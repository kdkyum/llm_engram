import argparse
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

QA_FIELDS = [
    ("bdate_q", "bdate_a"),
    ("bcity_q", "bcity_a"),
    ("university_q", "university_a"),
    ("major_q", "major_a"),
    ("employer_q", "employer_a"),
    ("company_city_q", "company_city_a")
]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned language models on QA tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model or HF model name")
    parser.add_argument("--model_type", type=str, default="gpt2", 
                        choices=["gpt2", "gpt-j", "gpt-neox", "llama"], 
                        help="Type of model to evaluate")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for evaluation")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional print statements")
    return parser.parse_args()

def create_multiple_choice_prompt(question, correct_answer, dataset, tokenizer=None, num_choices=4):
    """Create a multiple-choice prompt with one correct and three incorrect answers."""
    options = ["A", "B", "C", "D"]
    
    # Find field type from question
    qa_field = None
    for q_field, a_field in QA_FIELDS:
        if question.startswith(dataset[0][q_field].split()[0]):
            qa_field = a_field
            break
    
    if not qa_field:
        print(f"Could not identify question type for: {question}")
        return None, None
    
    # Get 3 random incorrect answers from the dataset
    incorrect_answers = []
    while len(incorrect_answers) < 3:
        random_idx = random.randint(0, len(dataset) - 1)
        random_answer = dataset[random_idx][qa_field]
        if random_answer != correct_answer and random_answer not in incorrect_answers:
            incorrect_answers.append(random_answer)
    
    # Create choices list with the correct answer at a random position
    choices = incorrect_answers.copy()
    correct_idx = random.randint(0, 3)
    choices.insert(correct_idx, correct_answer)
    
    # Format the multiple-choice prompt with a begin-of-text token if tokenizer is provided
    if tokenizer and hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
        prompt = f"{tokenizer.bos_token}{question}\n"
    else:
        # Default to using <|begin_of_text|> as string if tokenizer not available
        prompt = f"<|begin_of_text|>{question}\n"
        
    # Add the choices
    for i, (option, choice) in enumerate(zip(options, choices)):
        prompt += f"{option}. {choice}\n"
    prompt += "Answer: "
    
    return prompt, options[correct_idx]

def score_answers(model, tokenizer, prompts_and_answers, device, batch_size=16, debug=False):
    """Score the model's accuracy on multiple-choice questions with batching."""
    if not prompts_and_answers:
        return 0
    
    correct = 0
    options = ["A", "B", "C", "D"]
    option_tokens = {opt: tokenizer(opt, add_special_tokens=False).input_ids[0] for opt in options}
    
    # Print option tokens for debugging
    if debug:
        print("\n=== DEBUG: Option tokens ===")
        for opt, token_id in option_tokens.items():
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            print(f"Option {opt}: token_id={token_id}, token={token}")
    
    # Process in batches
    for i in tqdm(range(0, len(prompts_and_answers), batch_size)):
        batch_prompts = [p[0] for p in prompts_and_answers[i:i+batch_size]]
        batch_answers = [p[1] for p in prompts_and_answers[i:i+batch_size]]
        
        # Debug: print a sample prompt
        if debug and i == 0:
            print("\n=== DEBUG: Sample QA prompt ===")
            print(f"Prompt: {batch_prompts[0]}")
            print(f"Correct answer: {batch_answers[0]}")
            
            # Show tokenization of the prompt
            tokens = tokenizer.tokenize(batch_prompts[0])
            token_ids = tokenizer.encode(batch_prompts[0])
            print("\nTokenization:")
            for idx, (token, token_id) in enumerate(zip(tokens, token_ids)):
                print(f"Position {idx}: '{token}' (ID: {token_id})")
            
        # Tokenize all prompts in the batch
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        
        # Get all logits for the batch
        with torch.no_grad():
            outputs = model(**batch_inputs)
            # Get the last token logits for each sequence in the batch
            batch_logits = outputs.logits[:, -1, :]
        
        # For each item in the batch, find predicted answer
        for j, (logits, correct_answer) in enumerate(zip(batch_logits, batch_answers)):
            # Calculate scores for each option
            option_scores = {opt: logits[token_id].item() for opt, token_id in option_tokens.items()}
            
            # Get the model's prediction (option with highest score)
            model_answer = max(option_scores.items(), key=lambda x: x[1])[0]
            
            # Debug: print prediction details for first few examples
            if debug and i == 0 and j < 3:
                print(f"\n=== DEBUG: Prediction for example {j} ===")
                print(f"Prompt: {batch_prompts[j]}")
                print(f"Options scores: {option_scores}")
                print(f"Model prediction: {model_answer}")
                print(f"Correct answer: {correct_answer}")
                print(f"Correct? {'✓' if model_answer == correct_answer else '✗'}")
            
            # Check if the prediction is correct
            if model_answer == correct_answer:
                correct += 1
    
    accuracy = correct / len(prompts_and_answers)
    return accuracy

def evaluate_qa_by_type(model, tokenizer, test_dataset, args):
    """Evaluate QA accuracy by question type with batched processing."""
    results = {}
    overall_prompts = []
    
    # Collect prompts and answers for each QA type
    qa_type_prompts = {q_field: [] for q_field, _ in QA_FIELDS}
    
    # Debug: print an example record from test dataset
    if args.debug and len(test_dataset) > 0:
        print("\n=== DEBUG: Sample data record ===")
        sample_idx = 0
        sample = test_dataset[sample_idx]
        print(f"Record index: {sample_idx}")
        for field, value in sample.items():
            if field.endswith("_q") or field.endswith("_a"):
                print(f"{field}: {value}")
    
    for i in range(min(args.num_samples, len(test_dataset))):
        for q_field, a_field in QA_FIELDS:
            question = test_dataset[i][q_field]
            correct_answer = test_dataset[i][a_field]
            
            prompt, correct_option = create_multiple_choice_prompt(
                question, correct_answer, test_dataset, tokenizer
            )
            
            if prompt and correct_option:
                qa_type_prompts[q_field].append((prompt, correct_option))
    
    # Debug: print count of examples per question type
    if args.debug:
        print("\n=== DEBUG: Number of examples per question type ===")
        for q_field, prompts in qa_type_prompts.items():
            print(f"{q_field}: {len(prompts)} examples")
    
    # Calculate accuracy for each QA type using batched processing
    total_correct = 0
    total_samples = 0
    
    for q_field, prompts_and_answers in qa_type_prompts.items():
        print(f"Evaluating {q_field} questions...")
        
        # Print debug info for the first QA type only if in debug mode
        qa_debug = args.debug and q_field == QA_FIELDS[0][0]
        
        accuracy = score_answers(
            model, 
            tokenizer, 
            prompts_and_answers, 
            args.device, 
            batch_size=args.batch_size,
            debug=qa_debug
        )
        results[q_field] = accuracy
        print(f"{q_field} accuracy: {accuracy:.4f}")
        
        # Accumulate totals for overall accuracy
        total_correct += accuracy * len(prompts_and_answers)
        total_samples += len(prompts_and_answers)
    
    # Calculate overall accuracy from accumulated totals
    results["overall"] = total_correct / total_samples
    
    return results

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("minsungkim/bioS_v1")
    test_dataset = dataset["train"]  # Since there's only a train split, we'll sample from it
    
    # Load the model and tokenizer based on model type
    print(f"Loading model from {args.model_path}...")
    
    # Configure device and precision settings
    device_map = {"": args.device}
    torch_dtype = torch.float16 if args.fp16 else torch.float32
    
    # Load tokenizer and model based on model type
    if args.model_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            device_map=device_map,
            torch_dtype=torch_dtype
        )
    elif args.model_type == "gpt-j":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # GPT-J specific loading configurations
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
    elif args.model_type == "gpt-neox":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # GPT-NeoX specific loading configurations
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
    elif args.model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        # Llama models specific loading configurations
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Display model information
    print(f"Model type: {args.model_type}")
    print(f"Model name: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"FP16: {args.fp16}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Evaluate the model
    print(f"Evaluating model on {args.num_samples} samples...")
    results = evaluate_qa_by_type(model, tokenizer, test_dataset, args)
    
    # Print overall results
    print("\n===== RESULTS =====")
    for question_type, accuracy in results.items():
        print(f"{question_type}: {accuracy:.4f}")
    
    print(f"\nOverall Accuracy: {results['overall']:.4f}")

if __name__ == "__main__":
    main()