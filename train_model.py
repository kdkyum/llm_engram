#!/usr/bin/env python3

import os
import argparse
import random
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
import wandb
from tqdm import tqdm

# Import QA evaluation functions from evaluate_qa.py
from evaluate_qa import (
    create_multiple_choice_prompt, score_answers, 
    create_direct_answer_prompt, score_direct_answers, 
    QA_FIELDS
)

# Import TrainerCallback for QA evaluation
from transformers import TrainerCallback
from helpers import setup_offline_mode, get_device_fix_patch


class QAEvaluationCallback(TrainerCallback):
    """Custom callback for QA evaluation during training"""
    
    def __init__(self, trainer, eval_dataset, tokenizer, args):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.args = args
        self.eval_steps = 0
        
        # Track best scores for both evaluation formats
        self.best_option_accuracy = 0.0
        self.best_direct_accuracy = 0.0
        
        # Determine which format to use for the primary best model
        self.primary_format = args.primary_eval_format
        
        # Create directory for best model (only keep one best model based on primary format)
        self.best_model_path = os.path.join(args.output_dir, "best_model")
        os.makedirs(self.best_model_path, exist_ok=True)

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
            
            # Process results for both formats
            option_results = qa_results["option"]
            direct_results = qa_results["direct"]
            
            # Check if this is the best option-based model (just track scores)
            if option_results and "overall" in option_results:
                current_option_accuracy = option_results["overall"]
                if current_option_accuracy > self.best_option_accuracy:
                    self.best_option_accuracy = current_option_accuracy
                    print(f"\n=== New best option accuracy: {current_option_accuracy:.4f} (at epoch {state.epoch:.2f}) ===")
            
            # Check if this is the best direct answer model (just track scores)
            if direct_results and "overall" in direct_results:
                current_direct_accuracy = direct_results["overall"]
                if current_direct_accuracy > self.best_direct_accuracy:
                    self.best_direct_accuracy = current_direct_accuracy
                    print(f"\n=== New best direct answer accuracy: {current_direct_accuracy:.4f} (at epoch {state.epoch:.2f}) ===")
            
            # Only save the best model based on the primary evaluation format
            current_primary_accuracy = option_results["overall"] if self.primary_format == "option" else direct_results["overall"]
            best_primary_accuracy = self.best_option_accuracy if self.primary_format == "option" else self.best_direct_accuracy
            current_primary_results = option_results if self.primary_format == "option" else direct_results
            
            if current_primary_accuracy == best_primary_accuracy:
                print(f"\n=== New best model based on {self.primary_format} format! Accuracy: {current_primary_accuracy:.4f} (at epoch {state.epoch:.2f}) ===")
                
                # Save the best model
                print(f"Saving best model to {self.best_model_path}")
                
                # Save the model and tokenizer
                self.trainer.save_model(self.best_model_path)
                self.tokenizer.save_pretrained(self.best_model_path)
                
                # Save best model metrics
                with open(os.path.join(self.best_model_path, "best_metrics.txt"), "w") as f:
                    f.write(f"Primary format: {self.primary_format}\n")
                    f.write(f"Epoch: {state.epoch:.2f}\n")
                    f.write(f"Step: {state.global_step}\n")
                    f.write(f"Overall accuracy: {current_primary_results['overall']:.4f}\n")
                    for question_type, accuracy in current_primary_results.items():
                        if question_type != "overall":
                            f.write(f"{question_type}: {accuracy:.4f}\n")
            
            # Log to wandb with eval step to distinguish between evaluations
            wandb_metrics = {
                "eval/step": self.eval_steps,
                "eval/epoch": state.epoch,
            }
            
            # Add option format metrics
            if option_results and "overall" in option_results:
                wandb_metrics.update({
                    "eval/option_qa_overall_accuracy": option_results["overall"],
                    "eval/option_qa_bdate_accuracy": option_results["bdate_q"],
                    "eval/option_qa_bcity_accuracy": option_results["bcity_q"],
                    "eval/option_qa_university_accuracy": option_results["university_q"],
                    "eval/option_qa_major_accuracy": option_results["major_q"],
                    "eval/option_qa_employer_accuracy": option_results["employer_q"],
                    "eval/option_qa_company_city_accuracy": option_results["company_city_q"],
                    "eval/option_is_best_model": 1 if option_results["overall"] == self.best_option_accuracy else 0,
                })
            
            # Add direct format metrics
            if direct_results and "overall" in direct_results:
                wandb_metrics.update({
                    "eval/direct_qa_overall_accuracy": direct_results["overall"],
                    "eval/direct_qa_bdate_accuracy": direct_results["bdate_q"],
                    "eval/direct_qa_bcity_accuracy": direct_results["bcity_q"],
                    "eval/direct_qa_university_accuracy": direct_results["university_q"],
                    "eval/direct_qa_major_accuracy": direct_results["major_q"],
                    "eval/direct_qa_employer_accuracy": direct_results["employer_q"],
                    "eval/direct_qa_company_city_accuracy": direct_results["company_city_q"],
                    "eval/direct_is_best_model": 1 if direct_results["overall"] == self.best_direct_accuracy else 0,
                })
            
            # Add primary format indicator
            wandb_metrics.update({
                "eval/primary_format": 0 if self.primary_format == "option" else 1,
                "eval/primary_best_accuracy": best_primary_accuracy,
                "eval/saved_best_model": 1 if current_primary_accuracy == best_primary_accuracy else 0
            })
            
            wandb.log(wandb_metrics, step=state.global_step)
            
            # Print option format results
            if option_results and "overall" in option_results:
                print(f"\n===== OPTION FORMAT EVALUATION (Epoch {state.epoch:.2f}, Step {state.global_step}) =====")
                for question_type, accuracy in option_results.items():
                    if not question_type.startswith("mcq_"):  # Skip the MCQ-specific metrics
                        print(f"{question_type}: {accuracy:.4f}")
                print(f"Option Format Overall Accuracy: {option_results['overall']:.4f}")
                if option_results["overall"] == self.best_option_accuracy:
                    print("(Best option accuracy so far)")
            
            # Print direct format results
            if direct_results and "overall" in direct_results:
                print(f"\n===== DIRECT ANSWER FORMAT EVALUATION (Epoch {state.epoch:.2f}, Step {state.global_step}) =====")
                for question_type, accuracy in direct_results.items():
                    if not question_type.startswith("mcq_"):  # Skip the MCQ-specific metrics
                        print(f"{question_type}: {accuracy:.4f}")
                print(f"Direct Answer Format Overall Accuracy: {direct_results['overall']:.4f}")
                if direct_results["overall"] == self.best_direct_accuracy:
                    print("(Best direct answer accuracy so far)")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        # Return to training mode
        model.train()
        
    def evaluate_qa(self, model, tokenizer, eval_dataset, args):
        """Evaluate QA accuracy by question type using batched processing for both formats"""
        # Results dictionaries for both formats
        option_results = {}
        direct_results = {}
        
        # Determine which evaluation formats to use
        eval_option = args.eval_format in ["option", "both"]
        eval_direct = args.eval_format in ["direct", "both"]
        
        # Use a subset of eval dataset for faster evaluation during training
        subset_size = min(args.eval_samples, len(eval_dataset))
        indices = random.sample(range(len(eval_dataset)), subset_size)
        
        # Debug: Print information about the evaluation
        if args.debug:
            print(f"\n=== DEBUG: QA Evaluation ===")
            print(f"Evaluating on {subset_size} samples")
            print(f"Formats: {'option ' if eval_option else ''}{'direct' if eval_direct else ''}")
            print(f"Device: {model.device}")
        
        # Collect prompts and answers for each QA type, for both formats
        qa_type_option_prompts = {q_field: [] for q_field, _ in QA_FIELDS} if eval_option else {}
        qa_type_direct_prompts = {q_field: [] for q_field, _ in QA_FIELDS} if eval_direct else {}
        
        # First collect all prompts and answers
        for i in indices:
            for q_field, a_field in QA_FIELDS:
                question = eval_dataset[i][q_field]
                correct_answer = eval_dataset[i][a_field]
                
                if eval_option:
                    # Option-based prompts
                    option_prompt, option_correct = create_multiple_choice_prompt(
                        question, correct_answer, eval_dataset, a_field, with_fewshot=True
                    )
                    
                    if option_prompt and option_correct:
                        qa_type_option_prompts[q_field].append((option_prompt, option_correct))
                
                if eval_direct:
                    # Direct answer prompts
                    direct_prompt, direct_correct = create_direct_answer_prompt(
                        question, correct_answer, eval_dataset, a_field, with_fewshot=True
                    )
                    
                    if direct_prompt and direct_correct:
                        qa_type_direct_prompts[q_field].append((direct_prompt, direct_correct))
        
        # Debug: Print count of examples per question type for each format
        if args.debug:
            print("\n=== DEBUG: Number of examples per question type ===")
            if eval_option:
                for q_field, prompts in qa_type_option_prompts.items():
                    print(f"{q_field} (option format): {len(prompts)} examples")
            if eval_direct:
                for q_field, prompts in qa_type_direct_prompts.items():
                    print(f"{q_field} (direct format): {len(prompts)} examples")
        
        device = model.device
        
        # Calculate accuracy for option-based format
        if eval_option:
            option_total_correct = 0
            option_total_samples = 0
            
            for q_field, prompts_and_answers in qa_type_option_prompts.items():
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
                option_results[q_field] = accuracy
                
                # Add to totals for overall accuracy
                option_total_correct += accuracy * len(prompts_and_answers)
                option_total_samples += len(prompts_and_answers)
            
            # Calculate overall accuracy from accumulated results
            if option_total_samples > 0:
                option_results["overall"] = option_total_correct / option_total_samples
        
        # Calculate accuracy for direct answer format
        if eval_direct:
            direct_total_correct = 0
            direct_total_samples = 0
            
            for q_field, prompts_and_answers in qa_type_direct_prompts.items():
                # Only debug the first question type to avoid too much output
                debug_this_type = args.debug and q_field == QA_FIELDS[0][0]
                
                accuracy = score_direct_answers(
                    model, 
                    tokenizer, 
                    prompts_and_answers, 
                    device, 
                    batch_size=args.eval_batch_size,
                    debug=debug_this_type
                )
                direct_results[q_field] = accuracy
                
                # Add to totals for overall accuracy
                direct_total_correct += accuracy * len(prompts_and_answers)
                direct_total_samples += len(prompts_and_answers)
            
            # Calculate overall accuracy from accumulated results
            if direct_total_samples > 0:
                direct_results["overall"] = direct_total_correct / direct_total_samples
        
        # Combine results for return
        combined_results = {
            "option": option_results,
            "direct": direct_results
        }
        
        return combined_results


class BioDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, bio_field, max_length=256, mcq_percentage=0, 
                 mcq_with_bios=True, debug=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.bio_field = bio_field
        self.max_length = max_length
        self.debug = debug
        self.mcq_percentage = mcq_percentage
        self.mcq_with_bios = mcq_with_bios
        
        # Create a flat list of examples
        self.examples = []
        
        # First, add all regular bio examples
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
        
        # Track which samples have MCQ examples included for evaluation
        self.mcq_indices = set()
        
        # If MCQ percentage is greater than 0, add MCQ examples
        if mcq_percentage > 0:
            print(f"\nAdding {mcq_percentage}% of MCQ examples to training data...")
            
            # Calculate how many unique bioS samples to use for MCQ examples
            # Each bioS will contribute 6 MCQ examples (one for each question type)
            num_bio_samples = len(dataset)
            num_samples_to_select = int((mcq_percentage / 100) * num_bio_samples)
            
            print(f"Selecting {num_samples_to_select} bioS samples to extract MCQ examples from")
            
            # Get indices of samples to use for MCQ
            indices = list(range(num_bio_samples))
            random.shuffle(indices)
            selected_indices = indices[:num_samples_to_select]
            
            # Store the indices of samples used for MCQ for later evaluation comparisons
            self.mcq_indices = set(selected_indices)
            
            # Counter for added MCQ examples
            mcq_examples = []
            
            # For each selected sample, add all 6 question types
            for i in selected_indices:
                # For each question type in the current bioS sample
                for q_field, a_field in QA_FIELDS:
                    question = dataset[i][q_field]
                    correct_answer = dataset[i][a_field]
                    
                    # Always use fixed order for training examples
                    prompt, correct_option = create_multiple_choice_prompt(
                        question, correct_answer, dataset, a_field, with_fewshot=False
                    )
                    
                    if prompt and correct_option:
                        # Add MCQ example with full answer (not just the option)
                        mcq_text = prompt + correct_option + ". " + correct_answer
                        
                        # If we want to include the bio text with the MCQ
                        if mcq_with_bios:
                            bio_text = dataset[i][bio_field]
                            combined_text = f"{bio_text}\n\n{mcq_text}"
                            mcq_examples.append(combined_text)
                        else:
                            # Just use the MCQ text without the bio
                            mcq_examples.append(mcq_text)
            
            # Add MCQ examples to the list of training examples
            self.examples.extend(mcq_examples)
            
            # Calculate actual counts for reporting
            if mcq_with_bios:
                print(f"Added {len(mcq_examples)} MCQ examples with bioS text.")
                print(f"Total training examples: {len(self.examples)}")
                print(f"Each bioS sample contributed 6 MCQ examples (one for each question type).")
            else:
                print(f"Added {len(mcq_examples)} MCQ examples without bioS text.")
                print(f"Total training examples: {len(self.examples)}")
                print(f"For each bioS sample: 1 bioS + 6 MCQ examples = 7 total training examples.")
            
            # Print an example of MCQ for debugging
            if debug and mcq_examples:
                print("\n=== DEBUG: MCQ Example ===")
                print(mcq_examples[0])
        
        # Debug: print first few examples
        if debug:
            print("\n=== DEBUG: BioDataset Examples ===")
            print(f"Total examples: {len(self.examples)}")
            for i in range(min(3, len(self.examples))):
                print(f"\nExample {i}:")
                print(self.examples[i])
        # Shuffle the complete list of examples
        random.shuffle(self.examples)
    
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
        
        input_ids = inputs["input_ids"]
        inputs["labels"] = input_ids.copy()
        
        return inputs
    

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune language models on bioS dataset")
    parser.add_argument("--output_dir", type=str, default="./model-output", help="Directory to save model")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2-xl", help="Model to fine-tune")
    parser.add_argument("--model_type", type=str, default="gpt2", 
                    choices=["gpt2", "gpt-j", "gpt-neox", "llama", "olmo"], 
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
    parser.add_argument("--mcq_percentage", type=int, default=0, help="Percentage of MCQ evaluation examples to include in training data (0-100)")
    parser.add_argument("--mcq_with_bios", action="store_true", help="Include bioS text with MCQ examples (if not set, only MCQ text is used)")
    parser.add_argument("--shuffle_eval_choices", action="store_true", help="Shuffle MCQ choices during evaluation to test for overfitting")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional print statements")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode without syncing to wandb")
    # Add new arguments for evaluation format
    parser.add_argument("--eval_format", type=str, default="both", choices=["option", "direct", "both"], 
                      help="Evaluation format to use: option-based, direct answer, or both")
    parser.add_argument("--primary_eval_format", type=str, default="direct", choices=["option", "direct"],
                      help="Primary evaluation format to use for determining the best model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    if args.offline:
        setup_offline_mode()
    
    get_device_fix_patch()
    # Configure model output directory to include model name and training settings
    model_name_safe = args.model_name_or_path.replace("/", "-")
    
    # Add MCQ info to output directory if MCQ examples are used
    mcq_suffix = ""
    if args.mcq_percentage > 0:
        mcq_type = "withBioS" if args.mcq_with_bios else "noBioS"
        mcq_suffix = f"-mcq{args.mcq_percentage}p-{mcq_type}"
    
    # Calculate effective batch size for the model path
    n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    
    # Include evaluation format in output directory name
    eval_format_suffix = f"-eval{args.primary_eval_format}"
    
    # Include effective batch size in output directory name
    output_dir = os.path.join(args.output_dir, f"{model_name_safe}-{args.bio_field}{mcq_suffix}{eval_format_suffix}-ep{args.num_train_epochs}-bs{effective_batch_size}-samples{args.max_samples}-lr{args.learning_rate}-seed{args.seed}")
    args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Add MCQ and eval format info to run name
    mcq_run_suffix = ""
    if args.mcq_percentage > 0:
        mcq_type = "withBioS" if args.mcq_with_bios else "noBioS"
        mcq_run_suffix = f"-mcq{args.mcq_percentage}p-{mcq_type}"
    
    # Include evaluation format in run name
    eval_format_run_suffix = f"-eval{args.primary_eval_format}"
    
    # Include effective batch size in wandb run name
    run_name = args.wandb_run_name or f"{model_name_safe}-{args.bio_field}{mcq_run_suffix}{eval_format_run_suffix}-bs{effective_batch_size}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("minsungkim/bioS_v1")
    train_dataset = dataset["train"]
    
    # Import early to check for precomputed answers
    from evaluate_qa import create_multiple_choice_prompt, precomputed_answers
    
    # If precomputed answers are not available, compute them now
    if not precomputed_answers:
        print("Precomputed answers not found. Run precompute_answer_categories.py first for better performance.")
        print("Collecting unique answers from dataset (this might take a while)...")
        
        all_answers_by_field = {}
        for _, a_field in QA_FIELDS:
            # Get unique answers for this field
            unique_answers = set()
            for i in range(len(train_dataset)):
                answer = train_dataset[i][a_field]
                unique_answers.add(answer)
            
            # Store as list
            all_answers_by_field[a_field] = list(unique_answers)
            print(f"Collected {len(unique_answers)} unique answers for {a_field}")
        
        # Make this available for MCQ generation
        create_multiple_choice_prompt.all_answers_by_field = all_answers_by_field
        create_direct_answer_prompt.all_answers_by_field = all_answers_by_field
    else:
        print("Using precomputed answer categories")
    
    # Limit dataset size if max_samples is specified
    if args.max_samples is not None:
        print(f"Using first {args.max_samples} samples from dataset")
        train_dataset = train_dataset.select(range(min(args.max_samples, len(train_dataset))))
    
    # Load tokenizer and model based on model type
    print(f"Loading {args.model_type} model: {args.model_name_or_path}...")
    
    # Load tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        device_map="auto",
        use_cache=False,
        # Change dtype based on model type and fp16 flag (only if not using quantization)
        torch_dtype=torch.bfloat16 if args.fp16 and args.model_type in ["llama", "gpt-j", "gpt-neox", "olmo"] else torch.float32
    )
    
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
        elif args.model_type == "olmo":
            for param in model.model.embed_tokens.parameters():
                param.requires_grad = False
            for param in model.model.rotary_emb.parameters():
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
            elif args.model_type =="olmo":
                model.model.embed_tokens.register_forward_hook(embedding_forward_hook)
        
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
    tokenized_train = BioDataset(
        train_dataset, 
        tokenizer, 
        bio_field, 
        max_length=256,
        mcq_percentage=args.mcq_percentage,
        mcq_with_bios=args.mcq_with_bios,
        debug=args.debug
    )

    # Calculate total training steps considering multiple devices
    total_steps = (len(tokenized_train) // effective_batch_size) * args.num_train_epochs
    warmup_steps = int(total_steps * 0.02)
    print(f"Total training steps: {total_steps}, Warmup steps (2%): {warmup_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Configure training with updated mixed precision settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",  # Don't save model automatically
        eval_strategy="no",  # We'll do our own evaluation with the callback
        load_best_model_at_end=False,  # We'll manually save the best model in our callback
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        report_to="wandb",
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_steps=warmup_steps,
        fp16=args.fp16 and args.model_type == "gpt2",  # Use fp16 only for gpt2
        bf16=args.fp16 and args.model_type in ["llama", "gpt-j", "gpt-neox", "olmo"],
        half_precision_backend="auto",
        ddp_find_unused_parameters=False,
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
    
    # Get MCQ indices for evaluation comparison from the dataset
    mcq_indices = tokenized_train.mcq_indices if hasattr(tokenized_train, 'mcq_indices') else set()
   
    # Register the QA evaluation callback
    qa_eval_callback = QAEvaluationCallback(
        trainer=trainer,
        eval_dataset=train_dataset,
        tokenizer=tokenizer,
        args=args,
    )
    trainer.add_callback(qa_eval_callback)
    
    # Train model
    print("Training model...")
    trainer.train()

    # Print best model information
    print("\n===== TRAINING COMPLETE =====")
    print(f"Best option accuracy: {qa_eval_callback.best_option_accuracy:.4f}")
    print(f"Best direct answer accuracy: {qa_eval_callback.best_direct_accuracy:.4f}")
    print(f"Primary evaluation format: {args.primary_eval_format}")
    print(f"Best model saved at {qa_eval_callback.best_model_path} (based on {args.primary_eval_format} format)")
    
    # Log final best metrics to wandb for both formats
    final_metrics = {
        "best/option_overall_accuracy": qa_eval_callback.best_option_accuracy,
        "best/direct_overall_accuracy": qa_eval_callback.best_direct_accuracy,
        "best/primary_format": 0 if args.primary_eval_format == "option" else 1,
        "best/primary_accuracy": qa_eval_callback.best_option_accuracy if args.primary_eval_format == "option" else qa_eval_callback.best_direct_accuracy
    }
    
    # Add category-specific metrics for the primary format
    best_metrics_path = os.path.join(qa_eval_callback.best_model_path, "best_metrics.txt")
    if os.path.exists(best_metrics_path):
        with open(best_metrics_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Primary format:"):
                    # Store the primary format but don't try to convert to float
                    final_metrics["best/primary_format_str"] = line.split(":", 1)[1].strip()
                elif line.startswith("Epoch:"):
                    final_metrics["best/epoch"] = float(line.split(":", 1)[1].strip())
                elif line.startswith("Step:"):
                    final_metrics["best/step"] = int(line.split(":", 1)[1].strip())
                elif line.startswith("Overall accuracy:"):
                    final_metrics["best/overall_accuracy"] = float(line.split(":", 1)[1].strip())
                elif ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    if key not in ["Primary format"]:  # Skip already processed keys
                        format_prefix = "option" if args.primary_eval_format == "option" else "direct"
                        try:
                            final_metrics[f"best/{format_prefix}_{key}_accuracy"] = float(value.strip())
                        except ValueError:
                            print(f"Warning: Could not convert value '{value.strip()}' to float for key '{key}'")
    
    # Log all final metrics to wandb
    wandb.log(final_metrics)
    
    
if __name__ == "__main__":
    main()