#!/usr/bin/env python3

import os
import argparse
import random
import numpy as np
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    AdamW,
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


class BioDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, bio_field, max_length=256, debug=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.bio_field = bio_field
        self.max_length = max_length
        self.debug = debug
        
        # Create a flat list of examples
        self.examples = []
        if bio_field == "bioS_multi5_permutes":
            for idx in range(len(dataset)):
                # Each dataset item contains a list of 5 permutations
                # Each permutation is a complete biography text (not to be split further)
                for permutation in dataset[idx][bio_field]:
                    # Add each complete permutation as a separate example
                    if isinstance(permutation, str):
                        self.examples.append(permutation)
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
    parser.add_argument("--bio_field", type=str, default="bioS", help="Which bio field to use for training (bioS, bioS_fullname, or bioS_multi5_permutes)")
    parser.add_argument("--wandb_project", type=str, default="llm-engram", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of samples to use for QA evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size for QA evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to use from dataset")
    parser.add_argument("--freeze_embeddings", action="store_true", help="Freeze model embeddings")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--load_in_8bit", action="store_true", help="Enable 8-bit quantization for large models")
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
    
    # Configure model output directory to include model name and training settings
    model_name_safe = args.model_name_or_path.replace("/", "-")
    quantization_suffix = "-8bit" if args.load_in_8bit else ""
    output_dir = os.path.join(args.output_dir, f"{model_name_safe}-{args.bio_field}-ep{args.num_train_epochs}{quantization_suffix}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb with descriptive run name
    quantization_desc = "8bit" if args.load_in_8bit else ""
    run_name = args.wandb_run_name or f"{model_name_safe}-{args.bio_field}{'-' + quantization_desc if quantization_desc else ''}"
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
    if args.model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, add_bos_token=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization if requested
    quantization_config = None
    if args.load_in_8bit:
        print("Using 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    
    # Import PEFT for parameter-efficient fine-tuning
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    
    # Load model with appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        device_map="auto",
        use_cache=False,
        quantization_config=quantization_config,
        # Change dtype based on model type and fp16 flag (only if not using quantization)
        torch_dtype=torch.bfloat16 if args.fp16 and args.model_type in ["llama", "gpt-j", "gpt-neox"] and not args.load_in_8bit else torch.float32
    )
    
    # For quantized models, we need to use LoRA
    if args.load_in_8bit:
        print("Setting up LoRA for quantized model...")
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Define LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
    
    if args.freeze_embeddings:
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
            # Skip for LoRA models since we don't need to freeze embeddings
            if args.load_in_8bit:
                print("Skipping embedding freezing for LoRA-enabled model")
            else:
                print("Model structure for debugging:")
                for name, _ in model.named_modules():
                    if "embed" in name:
                        print(f"Found module: {name}")
                # Try different paths for embeddings based on model structure
                if hasattr(model, "base_model"):
                    if hasattr(model.base_model, "embed_tokens"):
                        for param in model.base_model.embed_tokens.parameters():
                            param.requires_grad = False
                        print("Froze embeddings at model.base_model.embed_tokens")
                    elif hasattr(model.base_model, "model") and hasattr(model.base_model.model, "embed_tokens"):
                        for param in model.base_model.model.embed_tokens.parameters():
                            param.requires_grad = False
                        print("Froze embeddings at model.base_model.model.embed_tokens")
                elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                    for param in model.model.embed_tokens.parameters():
                        param.requires_grad = False
                    print("Froze embeddings at model.model.embed_tokens")
                else:
                    print("Warning: Could not find embedding layer for Llama model")
                
                # Freeze LM head if it exists
                if hasattr(model, "lm_head"):
                    for param in model.lm_head.parameters():
                        param.requires_grad = False
                    print("Froze lm_head layer")
        elif args.model_type == "gpt2":
            for param in model.transformer.wte.parameters():
                param.requires_grad = False
            for param in model.transformer.wpe.parameters():
                param.requires_grad = False
            for param in model.lm_head.parameters():
                param.requires_grad = False

    if args.gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        if args.freeze_embeddings:
            def embedding_forward_hook(module, input, output):
                """Force embedding output to require gradients, even if embeddings are frozen."""
                output.requires_grad_(True)
                return output
            
            # Register forward hook based on model type
            if args.model_type == "gpt-j":
                model.transformer.wte.register_forward_hook(embedding_forward_hook)
            elif args.model_type == "llama":
                # Skip for LoRA models since embedding handling is different
                if args.load_in_8bit:
                    print("Skipping embedding hook for LoRA-enabled model")
                else:
                    # Try different paths for embeddings based on model structure
                    if hasattr(model, "base_model"):
                        if hasattr(model.base_model, "embed_tokens"):
                            model.base_model.embed_tokens.register_forward_hook(embedding_forward_hook)
                            print("Registered hook at model.base_model.embed_tokens")
                        elif hasattr(model.base_model, "model") and hasattr(model.base_model.model, "embed_tokens"):
                            model.base_model.model.embed_tokens.register_forward_hook(embedding_forward_hook)
                            print("Registered hook at model.base_model.model.embed_tokens")
                    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                        model.model.embed_tokens.register_forward_hook(embedding_forward_hook)
                        print("Registered hook at model.model.embed_tokens")
                    else:
                        print("Warning: Could not find embedding layer for hook in Llama model")
            elif args.model_type == "gpt2":
                model.transformer.wte.register_forward_hook(embedding_forward_hook)
            elif args.model_type == "gpt-neox":
                model.gpt_neox.embed_in.register_forward_hook(embedding_forward_hook)
        
        model.gradient_checkpointing_enable()
    
    # Print trainable parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
    
    # Tokenize data
    print("Tokenizing data...")
    bio_field = args.bio_field
    tokenized_train = BioDataset(train_dataset, tokenizer, bio_field, debug=args.debug)
    
    # Configure training with updated mixed precision settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",  # Don't save model every epoch
        eval_strategy="no",  # We'll do our own evaluation with the callback
        load_best_model_at_end=False,  # We'll manually save the best model in our callback
        report_to="wandb",
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="constant",  # Use constant learning rate
        # Update mixed precision settings
        fp16=args.fp16 and args.model_type == "gpt2" and not args.load_in_8bit,  # Use fp16 only for gpt2
        bf16=args.fp16 and args.model_type in ["llama", "gpt-j", "gpt-neox"] and not args.load_in_8bit,  # Use bf16 for llama, gpt-j, and gpt-neox
        half_precision_backend="auto"
    )
            
    # Create a custom data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to False for causal LM (not masked LM)
    )
    
    model.train()
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train,
    )
    
    # Train model
    print("Training model...")
    trainer.train()

    # Save model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Final evaluation using MCQ format on the full training dataset
    print("Performing final QA evaluation using MCQ format...")
    model.eval()
    # Add custom QA evaluation callback
    qa_eval_callback = QAEvaluationCallback(
        trainer=trainer,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        args=args
    )
    final_results = qa_eval_callback.evaluate_qa(model, tokenizer, eval_dataset, args)
    
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