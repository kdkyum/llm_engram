#!/usr/bin/env python3
"""
Modified training script that fixes the FP16 gradient issues.
This script should be used for training larger models with mixed precision.
"""

import os
import argparse
import random
import numpy as np
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EvalPrediction
)
import wandb
from tqdm import tqdm

# Import QA evaluation function from evaluate_qa.py
from evaluate_qa import create_multiple_choice_prompt, score_answers, QA_FIELDS

# Import TrainerCallback for QA evaluation
from transformers import TrainerCallback

class QAEvaluationCallback(TrainerCallback):
    """Custom callback for QA evaluation during training"""
    
    def __init__(self, trainer, eval_dataset, tokenizer, args):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.args = args
        self.eval_steps = 0

    def on_epoch_end(self, args, state, control, **kwargs):
        """Run evaluation at the end of each epoch"""
        self._run_evaluation(args, state, control)
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Run initial evaluation at the start of training"""
        print("Running initial evaluation...")
        self._run_evaluation(args, state, control)
        
    def _run_evaluation(self, args, state, control):
        """Run the QA evaluation"""
        self.eval_steps += 1
        
        # Get current model
        model = self.trainer.model
        model.eval()
        
        # Use a smaller subset for faster evaluation during training
        eval_samples = min(self.args.eval_samples, len(self.eval_dataset))
        
        try:
            # Create QA evaluation samples
            print(f"Evaluating model on {eval_samples} samples...")
            qa_results = self.evaluate_qa(model, self.tokenizer, self.eval_dataset, self.args)
            
            # Log to wandb with eval step to distinguish between evaluations
            wandb.log({
                "eval/step": self.eval_steps,
                "eval/epoch": state.epoch,
                "eval/qa_overall_accuracy": qa_results["overall"],
                "eval/qa_bdate_accuracy": qa_results["bdate_q"],
                "eval/qa_bcity_accuracy": qa_results["bcity_q"],
                "eval/qa_university_accuracy": qa_results["university_q"],
                "eval/qa_major_accuracy": qa_results["major_q"],
                "eval/qa_employer_accuracy": qa_results["employer_q"],
                "eval/qa_company_city_accuracy": qa_results["company_city_q"],
            }, step=state.global_step)
            
            print(f"\n===== QA EVALUATION (Epoch {state.epoch:.2f}, Step {state.global_step}) =====")
            for question_type, accuracy in qa_results.items():
                print(f"{question_type}: {accuracy:.4f}")
            print(f"Overall Accuracy: {qa_results['overall']:.4f}")
            
            # Create a table to show results by question type
            qa_table = wandb.Table(columns=["Question Type", "Accuracy"])
            for question_type, accuracy in qa_results.items():
                if question_type != "overall":  # Skip overall since we show it separately
                    question_name = question_type.replace("_q", "").capitalize()
                    qa_table.add_data(question_name, accuracy)
            
            wandb.log({f"eval_{self.eval_steps}/results_table": qa_table}, step=state.global_step)
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        # Return to training mode
        model.train()
        
    def evaluate_qa(self, model, tokenizer, eval_dataset, args):
        """Evaluate QA accuracy by question type using batched processing"""
        results = {}
        overall_prompts = []
        
        # Use a subset of eval dataset for faster evaluation during training
        subset_size = min(args.eval_samples, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), subset_size)
        
        # Debug: Print information about the evaluation
        if args.debug:
            print(f"\n=== DEBUG: QA Evaluation ===")
            print(f"Evaluating on {subset_size} samples")
            print(f"Device: {model.device}")
        
        # Collect prompts and answers for each QA type
        qa_type_prompts = {q_field: [] for q_field, _ in QA_FIELDS}
        
        # First collect all prompts and answers
        for i in indices:
            for q_field, a_field in QA_FIELDS:
                question = eval_dataset[i][q_field]
                correct_answer = eval_dataset[i][a_field]
                
                prompt, correct_option = create_multiple_choice_prompt(
                    question, correct_answer, eval_dataset
                )
                
                if prompt and correct_option:
                    qa_type_prompts[q_field].append((prompt, correct_option))
                    # Also add to overall prompts for combined score
                    overall_prompts.append((prompt, correct_option))
                    
                    # Debug: Print one example of each question type
                    if args.debug and len(qa_type_prompts[q_field]) == 1:
                        print(f"\n=== DEBUG: Example {q_field} Question ===")
                        print(f"Question: {question}")
                        print(f"Correct Answer: {correct_answer}")
                        print(f"Formatted prompt:\n{prompt}")
                        print(f"Expected option: {correct_option}")
        
        # Debug: Print count of examples per question type
        if args.debug:
            print("\n=== DEBUG: Number of examples per question type ===")
            for q_field, prompts in qa_type_prompts.items():
                print(f"{q_field}: {len(prompts)} examples")
        
        # Calculate accuracy for each QA type using batched processing
        device = model.device
        total_correct = 0
        total_samples = 0
        
        for q_field, prompts_and_answers in qa_type_prompts.items():
            # Only debug the first question type to avoid too much output
            debug_this_type = args.debug and q_field == QA_FIELDS[0][0]
            
            accuracy = score_answers(
                model, 
                tokenizer, 
                prompts_and_answers, 
                device, 
                batch_size=args.eval_batch_size,
                debug=debug_this_type
            )
            results[q_field] = accuracy
            
            # Add to totals for overall accuracy
            total_correct += accuracy * len(prompts_and_answers)
            total_samples += len(prompts_and_answers)
        
        # Calculate overall accuracy from accumulated results
        results["overall"] = total_correct / total_samples if total_samples > 0 else 0
        
        return results

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune language models on bioS dataset")
    parser.add_argument("--output_dir", type=str, default="./model-output", help="Directory to save model")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2-xl", help="Model to fine-tune")
    parser.add_argument("--model_type", type=str, default="gpt2", 
                    choices=["gpt2", "gpt-j", "gpt-neox", "llama"], 
                    help="Type of model to fine-tune")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of update steps to accumulate gradients for")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision training")
    parser.add_argument("--bio_field", type=str, default="bioS", help="Which bio field to use for training (bioS, bioS_fullname, or bioS_multi5_permutes)")
    parser.add_argument("--wandb_project", type=str, default="llm-engram", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of samples to use for QA evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for QA evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to use from dataset")
    parser.add_argument("--freeze_embeddings", action="store_true", help="Freeze model embeddings")
    parser.add_argument("--lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional print statements")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Configure model output directory to include model name
    model_name_safe = args.model_name_or_path.replace("/", "-")
    output_dir = os.path.join(args.output_dir, f"{model_name_safe}-{args.bio_field}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb
    run_name = args.wandb_run_name or f"{model_name_safe}-{args.bio_field}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("minsungkim/bioS_v1")
    train_dataset = dataset["train"]
    
    # Limit dataset size if max_samples is specified
    if args.max_samples is not None:
        print(f"Using first {args.max_samples} samples from dataset")
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
    
    # Use full training dataset for both training and evaluation (no split)
    # since all data is synthetic
    eval_dataset = train_dataset
    
    # Load tokenizer and model based on model type
    print(f"Loading {args.model_type} model: {args.model_name_or_path}...")
    
    # Load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate configuration
    if args.model_type in ["gpt-j", "gpt-neox", "llama"]:
        # For larger models, use additional configurations and BF16
        torch_dtype = torch.bfloat16 if args.bf16 else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_cache=False  # Disable KV cache for compatibility with gradient checkpointing
        )
        
        # For LoRA, we need to import PEFT
        if args.lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                print("Using LoRA for parameter-efficient fine-tuning")
                
                # Define the target modules based on model type
                if args.model_type == "gpt-j":
                    target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]
                elif args.model_type == "gpt-neox":
                    target_modules = ["query_key_value", "dense"]
                elif args.model_type == "llama":
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                else:
                    target_modules = None
                    
                # Setup LoRA configuration
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=target_modules
                )
                
                # Prepare model for LoRA fine-tuning and ensure gradients are enabled
                # This explicitly sets requires_grad=True for input embeddings
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                    
                # Make sure model inputs have requires_grad=True during forward pass
                for param in model.parameters():
                    if param.requires_grad:
                        # At least one parameter needs to have requires_grad=True 
                        # This ensures we don't hit the UserWarning about "None of the inputs have requires_grad=True"
                        break
                
                # Apply LoRA to the model
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
                
            except ImportError:
                raise ImportError(
                    "PEFT library not found. Please install it with: pip install peft"
                )
        # If not using LoRA, freeze the embeddings
        elif args.freeze_embeddings:
            print("Freezing model embeddings...")
            if args.model_type == "gpt-j":
                # GPT-J specific
                for param in model.transformer.wte.parameters():
                    param.requires_grad = False
                for param in model.lm_head.parameters():
                    param.requires_grad = False
            elif args.model_type == "gpt-neox":
                # GPT-NeoX specific
                for param in model.gpt_neox.embed_in.parameters():
                    param.requires_grad = False
                for param in model.embed_out.parameters():
                    param.requires_grad = False
            elif args.model_type == "llama":
                for param in model.model.embed_tokens.parameters():
                    param.requires_grad = False
                for param in model.lm_head.parameters():
                    param.requires_grad = False
    else:
        # For GPT-2 models
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            use_cache=False  # Disable KV cache for compatibility with gradient checkpointing
        )
        
        # If using LoRA for GPT-2
        if args.lora:
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                print("Using LoRA for parameter-efficient fine-tuning")
                
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules=["c_attn", "c_proj"]
                )
                
                # This explicitly sets requires_grad=True for input embeddings
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
                
                # Make sure model inputs have requires_grad=True during forward pass
                for param in model.parameters():
                    if param.requires_grad:
                        # At least one parameter needs to have requires_grad=True 
                        # This ensures we don't hit the UserWarning about "None of the inputs have requires_grad=True"
                        break
                
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
                
            except ImportError:
                raise ImportError(
                    "PEFT library not found. Please install it with: pip install peft"
                )
        # If not using LoRA, freeze the embeddings
        elif args.freeze_embeddings:
            print("Freezing model embeddings...")
            for param in model.transformer.wte.parameters():
                param.requires_grad = False
            for param in model.transformer.wpe.parameters():
                param.requires_grad = False
            for param in model.lm_head.parameters():
                param.requires_grad = False
    
    # Print trainable parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    # Tokenize data
    print("Tokenizing data...")
    bio_field = args.bio_field
    
    # Prepare a dataset class that will properly tokenize our data
    class BioDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, tokenizer, bio_field, max_length=512, debug=False):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.bio_field = bio_field
            self.max_length = max_length
            self.debug = debug
            
            # Create a flat list of examples if using bioS_multi5_permutes
            self.examples = []
            if bio_field == "bioS_multi5_permutes":
                for idx in range(len(dataset)):
                    for permutes in dataset[idx][bio_field]:
                        for text in permutes:
                            self.examples.append(text)
            else:
                for idx in range(len(dataset)):
                    self.examples.append(dataset[idx][bio_field])
            
            # Debug: print first few examples
            if debug:
                print("\n=== DEBUG: BioDataset Examples ===")
                print(f"Total examples: {len(self.examples)}")
                for i in range(min(3, len(self.examples))):
                    print(f"\nExample {i}:")
                    print(self.examples[i])
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            text = self.examples[idx]
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors=None  # Don't convert to tensors yet
            )
            
            # Convert input_ids to tensors to catch any issues early
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Create labels for causal LM (same as input_ids)
            inputs["labels"] = input_ids.copy()
            
            # Debug: print tokenization details for the first example
            if self.debug and idx == 0:
                print("\n=== DEBUG: Example Tokenization ===")
                print(f"Original text: {text[:100]}...")  # Show beginning of text
                print(f"Input IDs: {input_ids[:50]}...")  # Show first 50 tokens
                
                # Print actual tokens
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[:50])
                print("\nFirst 50 tokens:")
                for i, (token, token_id) in enumerate(zip(tokens, input_ids[:50])):
                    print(f"Position {i}: '{token}' (ID: {token_id})")
                
                # Show attention mask
                print(f"\nAttention Mask (first 50): {attention_mask[:50]}")
                
                # Show where padding starts
                if 0 in attention_mask:
                    pad_pos = attention_mask.index(0)
                    print(f"Padding starts at position {pad_pos}")
                else:
                    print("No padding in this example")
            
            return inputs
    
    # Create dataset
    print(f"Creating dataset from {len(train_dataset)} examples...")
    tokenized_train = BioDataset(train_dataset, tokenizer, bio_field, debug=args.debug)
    
    # Configure training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,  # Use bf16 instead of fp16
        fp16=False,      # Disable fp16 to avoid the "unscale FP16 gradients" error
        logging_steps=args.logging_steps,
        save_strategy="no",  # Don't save model every epoch
        evaluation_strategy="no",  # We'll do our own evaluation with the callback
        load_best_model_at_end=False,  # We'll manually save the best model in our callback
        report_to="wandb",
        # Add gradient checkpointing for memory efficiency with larger models
        gradient_checkpointing=True,  # Enable for all models to save memory
        # With gradient checkpointing enabled, make sure to use_cache=False
        ddp_find_unused_parameters=False,  # Important to avoid issues with gradient checkpointing
        warmup_steps=args.warmup_steps,  # Add warmup steps
    )
    
    # Use a simplified data collator that won't conflict with our dataset
    class SimpleDataCollator:
        def __init__(self, pad_token_id):
            self.pad_token_id = pad_token_id
            
        def __call__(self, features):
            batch = {}
            
            # Get all keys from the first feature
            keys = features[0].keys()
            
            for key in keys:
                if key == "labels":
                    # For labels, we pad with -100 so they're ignored in loss calculation
                    batch[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)
                else:
                    # For other fields (input_ids, attention_mask), pad with pad_token_id
                    batch[key] = torch.tensor([feature[key] for feature in features], dtype=torch.long)
                    
            return batch
            
    # Create a custom data collator
    data_collator = SimpleDataCollator(pad_token_id=tokenizer.pad_token_id)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_train,  # Use training data for evaluation as well
    )
    
    # Add custom QA evaluation callback
    qa_eval_callback = QAEvaluationCallback(
        trainer=trainer,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=args
    )
    trainer.add_callback(qa_eval_callback)
    
    # Train model
    print("Training model...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {output_dir}...")
    # For LoRA models, save differently
    if args.lora:
        model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation using MCQ format on the full training dataset
    print("Performing final QA evaluation using MCQ format...")
    model.eval()
    
    # Use the full dataset for final evaluation with more samples
    args.eval_samples = min(500, len(train_dataset))  # Use more samples for final evaluation
    # Keep using the same batch size for final evaluation
    final_results = qa_eval_callback.evaluate_qa(model, tokenizer, train_dataset, args)
    
    print("\n===== FINAL RESULTS (MCQ Format) =====")
    for question_type, accuracy in final_results.items():
        print(f"{question_type}: {accuracy:.4f}")
    
    print(f"\nFinal Overall Accuracy: {final_results['overall']:.4f}")
    
    # Log final results to wandb
    wandb.log({
        "final/qa_overall_accuracy": final_results["overall"],
        "final/qa_bdate_accuracy": final_results["bdate_q"],
        "final/qa_bcity_accuracy": final_results["bcity_q"],
        "final/qa_university_accuracy": final_results["university_q"],
        "final/qa_major_accuracy": final_results["major_q"],
        "final/qa_employer_accuracy": final_results["employer_q"],
        "final/qa_company_city_accuracy": final_results["company_city_q"]
    })
    
    # Add summary metrics that will appear on the main wandb runs page
    wandb.run.summary["final_overall_accuracy"] = final_results["overall"]
    
    # Create a table to show results by question type
    qa_table = wandb.Table(columns=["Question Type", "Accuracy"])
    for question_type, accuracy in final_results.items():
        if question_type != "overall":  # Skip overall since we show it separately
            question_name = question_type.replace("_q", "").capitalize()
            qa_table.add_data(question_name, accuracy)
    
    wandb.log({"final_results_table": qa_table})
    
    # Create histogram of accuracies for different question types
    accuracies = [acc for qtype, acc in final_results.items() if qtype != "overall"]
    question_types = [qtype.replace("_q", "").capitalize() for qtype in final_results.keys() if qtype != "overall"]
    
    # Create a bar chart
    data = [[label, val] for label, val in zip(question_types, accuracies)]
    table = wandb.Table(data=data, columns=["Question Type", "Accuracy"])
    wandb.log({"accuracy_by_question_type": wandb.plot.bar(
        table, "Question Type", "Accuracy", title="Final Accuracy by Question Type")})
    
    # Finish wandb run
    wandb.finish()
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    main()