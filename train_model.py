#!/usr/bin/env python3

import os
import argparse
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

# Import QA evaluation functions from evaluate_qa.py
from evaluate_qa import create_direct_answer_prompt, QA_FIELDS
from helpers import setup_offline_mode, get_device_fix_patch, QAEvaluationCallback, freeze_model_embeddings, enable_gradient_checkpointing, optim_configs
from dataset import BioDataset


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
    parser.add_argument("--eval_format", type=str, default="direct", choices=["option", "direct", "both"], 
                      help="Evaluation format to use: option-based, direct answer, or both")
    parser.add_argument("--primary_eval_format", type=str, default="direct", choices=["option", "direct"],
                      help="Primary evaluation format to use for determining the best model")
    # Add weight_decay argument
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight decay value")
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
    freeze_suffix = "" if not args.freeze_embeddings else "-freeze"
    # Format weight decay for the path
    # Use optim_configs value if weight_decay is None
    model_id = args.model_name_or_path
    default_weight_decay = optim_configs.get(model_id, {"weight_decay": 0.01})["weight_decay"]
    weight_decay_value = args.weight_decay if args.weight_decay is not None else default_weight_decay
    weight_decay_str = f"-wd{weight_decay_value}"
    
    output_dir = os.path.join(args.output_dir, f"{model_name_safe}-{args.bio_field}{mcq_suffix}{eval_format_suffix}-ep{args.num_train_epochs}-bs{effective_batch_size}-samples{args.max_samples}-lr{args.learning_rate}{weight_decay_str}{freeze_suffix}-seed{args.seed}")
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
        freeze_model_embeddings(model, args)

    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model, args)
    
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
    # Get optimizer parameters from optim_configs dictionary
    model_id = args.model_name_or_path
    optim_params = optim_configs.get(model_id, {
        "beta1": 0.9,
        "beta2": 0.95,
        "epsilon": 1e-8,
        "weight_decay": 0.01,
    })  # Default values if model not found
    
    if args.debug:
        print(f"Using optimizer parameters for {model_id}: {optim_params}")
    
    # Override weight decay if provided via command line
    if args.weight_decay is not None:
        optim_params["weight_decay"] = args.weight_decay
        if args.debug:
            print(f"Overriding weight decay with command line value: {args.weight_decay}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",  # Don't save model automatically
        eval_strategy="no",  # Evaluate at regular intervals
        load_best_model_at_end=False,  # We'll manually save the best model in our callback
        adam_beta1=optim_params["beta1"],
        adam_beta2=optim_params["beta2"],
        adam_epsilon=optim_params["epsilon"],
        weight_decay=optim_params["weight_decay"],
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