import argparse
import random
import numpy as np
import torch
import json
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from rouge_score import rouge_scorer

QA_FIELDS = [
    ("bdate_q", "bdate_a"),
    ("bcity_q", "bcity_a"),
    ("university_q", "university_a"),
    ("major_q", "major_a"),
    ("employer_q", "employer_a"),
    ("company_city_q", "company_city_a")
]

# Path to precomputed unique answers
UNIQUE_ANSWERS_FILE = "unique_answers.json"

# Load precomputed unique answers if available
def load_unique_answers():
    if os.path.exists(UNIQUE_ANSWERS_FILE):
        try:
            with open(UNIQUE_ANSWERS_FILE, 'r') as f:
                unique_answers = json.load(f)
            print(f"Loaded precomputed unique answers from {UNIQUE_ANSWERS_FILE}")
            return unique_answers
        except Exception as e:
            print(f"Error loading unique answers: {e}")
    return None

# Global variable to store precomputed answers
precomputed_answers = load_unique_answers()

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
    parser.add_argument("--eval_format", type=str, default="both", choices=["option", "direct", "both"], 
                        help="Evaluation format: option-based, direct answer, or both")
    return parser.parse_args()

def create_multiple_choice_prompt(question, correct_answer, dataset, a_field, with_fewshot=True):
    """Create a multiple-choice prompt where the model should respond with the option number."""
    options = ["1", "2", "3", "4"]
    
    all_field_answers = precomputed_answers[a_field]
        
    # Filter out the correct answer
    available_options = [a for a in all_field_answers if a != correct_answer]
    
    random.shuffle(available_options)
    
    # Take first 3 as incorrect answers
    incorrect_answers = available_options[:3]
    
    # If we somehow don't have enough options, fall back to old method
    if len(incorrect_answers) < 3:
        print(f"Warning: Not enough unique answers for {a_field}, using fallback method")
        # Fallback to original implementation
        incorrect_answers = []
        while len(incorrect_answers) < 3:
            random_idx = random.randint(0, len(dataset) - 1)
            random_answer = dataset[random_idx][a_field]
            if random_answer != correct_answer and random_answer not in incorrect_answers:
                incorrect_answers.append(random_answer)
   
    # Create choices list with the correct answer at a random position
    choices = incorrect_answers.copy()
    correct_idx = random.randint(0, 3)
    choices.insert(correct_idx, correct_answer)
    
    # Updated few-shot examples to demonstrate direct answer format
    few_shot_examples = """Example 1:
When was Will Smith born?
1. January 8, 1987
2. April 23, 1975
3. June 12, 1990
4. September 25, 1968
Answer: 4. September 25, 1968

Example 2:
Where was Cameron Diaz born?
1. Chicago, Illinois
2. Portland, Oregon
3. San Diego, California
4. Seattle, Washington
Answer: 3. San Diego, California

Example 3:
What company does Sergey Brin work for?
1. Microsoft
2. Google
3. Amazon
4. Apple
Answer: 2. Google

Example 4:
"""
    if with_fewshot:
        prompt = few_shot_examples + f"{question}\n"
    else:
        prompt = f"{question}\n"
        
    # Add the choices
    for i, (option, choice) in enumerate(zip(options, choices)):
        prompt += f"{option}. {choice}\n"
    prompt += "Answer:"

    return prompt, options[correct_idx]   # Return the answer option letter

def create_direct_answer_prompt(question, correct_answer, dataset, a_field, with_fewshot=True):
    """Create a multiple-choice prompt where the model should respond with the actual answer."""
    options = ["1", "2", "3", "4"]
    
    all_field_answers = precomputed_answers[a_field]
        
    # Filter out the correct answer
    available_options = [a for a in all_field_answers if a != correct_answer]
    
    random.shuffle(available_options)
    
    # Take first 3 as incorrect answers
    incorrect_answers = available_options[:3]
    
    # If we somehow don't have enough options, fall back to old method
    if len(incorrect_answers) < 3:
        print(f"Warning: Not enough unique answers for {a_field}, using fallback method")
        # Fallback to original implementation
        incorrect_answers = []
        while len(incorrect_answers) < 3:
            random_idx = random.randint(0, len(dataset) - 1)
            random_answer = dataset[random_idx][a_field]
            if random_answer != correct_answer and random_answer not in incorrect_answers:
                incorrect_answers.append(random_answer)
   
    # Create choices list with the correct answer at a random position
    choices = incorrect_answers.copy()
    correct_idx = random.randint(0, 3)
    choices.insert(correct_idx, correct_answer)
    
    # Updated few-shot examples to demonstrate direct answer format
    few_shot_examples = """Example 1:
When was Will Smith born?
1. January 8, 1987
2. April 23, 1975
3. June 12, 1990
4. September 25, 1968
Answer: September 25, 1968

Example 2:
Where was Cameron Diaz born?
1. Chicago, Illinois
2. Portland, Oregon
3. San Diego, California
4. Seattle, Washington
Answer: San Diego, California

Example 3:
What company does Sergey Brin work for?
1. Microsoft
2. Google
3. Amazon
4. Apple
Answer: Google

Example 4:
"""
    if with_fewshot:
        prompt = few_shot_examples + f"{question}\n"
    else:
        prompt = f"{question}\n"
        
    # Add the choices
    for i, (option, choice) in enumerate(zip(options, choices)):
        prompt += f"{option}. {choice}\n"
    prompt += "Answer:"
    
    return prompt, correct_answer  # Return the actual answer, not the option letter

def score_answers(model, tokenizer, prompts_and_answers, device, batch_size=16, debug=False):
    """Score the model's accuracy on multiple-choice questions by comparing logits of candidate options."""
    if not prompts_and_answers:
        return 0
    
    correct = 0
    total = len(prompts_and_answers)
    options = ["1", "2", "3", "4"]

    is_llama_tokenizer = len(tokenizer.encode(" 1", add_special_tokens=False)) > 1

    if not is_llama_tokenizer:
        options = [" 1", " 2", " 3", " 4"]

    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[-1] for opt in options]
    
    # Process in batches
    for i in tqdm(range(0, total, batch_size)):
        if is_llama_tokenizer:
            batch_prompts = [p[0] + " " for p in prompts_and_answers[i:i+batch_size]]
        else:
            batch_prompts = [p[0] for p in prompts_and_answers[i:i+batch_size]]
        batch_answers = [p[1] for p in prompts_and_answers[i:i+batch_size]]
        
        # Tokenize all prompts in the batch
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        
        # Debug: print a sample prompt
        if debug and i == 0:
            print("\n=== DEBUG: Sample QA prompt (option-based) ===")
            sample_prompt = tokenizer.decode(batch_inputs['input_ids'][0])
            print(f"Prompt: {sample_prompt}")
            print(f"Correct answer: {batch_answers[0]}")

        # Forward pass to get logits
        with torch.no_grad():
            outputs = model(batch_inputs.input_ids, attention_mask=batch_inputs.attention_mask)
            logits = outputs.logits
        
        # For each example in batch, get the option with highest probability
        for j, (logit, correct_answer) in enumerate(zip(logits, batch_answers)):
            # Get the last token logits (for the position right after "Answer:")
            last_token_logits = logit[-1, :]
            
            # Get probabilities for each option token
            option_probs = {opt: last_token_logits[token_id].item() for opt, token_id in zip(options, option_token_ids)}
            
            # Get the option with the highest probability
            predicted_option = max(option_probs.items(), key=lambda x: x[1])[0]
            
            is_correct = (predicted_option.strip() == correct_answer.strip())
            
            if is_correct:
                correct += 1
            
            # Debug: print prediction details
            if debug:
                print(f"\n=== DEBUG: Prediction for example {i*batch_size + j} ===")
                print(f"Prompt ending: ...{batch_prompts[j][-50:]}")
                print(f"Option logits: {option_probs}")
                print(f"Predicted option: {predicted_option}")
                print(f"Correct option: {correct_answer}")
                print(f"Correct? {'✓' if is_correct else '✗'}")
                print("-" * 50)
    
    accuracy = correct / total
    return accuracy

def score_direct_answers(model, tokenizer, prompts_and_answers, device, batch_size=16, debug=False):
    """Score the model's accuracy on multiple-choice questions using direct answer generation."""
    if not prompts_and_answers:
        return 0
    
    correct = 0
    total = len(prompts_and_answers)
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    
    # Process in batches
    for i in tqdm(range(0, total, batch_size)):
        batch_prompts = [p[0] for p in prompts_and_answers[i:i+batch_size]]
        batch_answers = [p[1] for p in prompts_and_answers[i:i+batch_size]]
        
        # Tokenize all prompts in the batch
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(device)
        
        # Debug: print a sample prompt
        if debug and i == 0:
            print("\n=== DEBUG: Sample QA prompt (direct answer) ===")
            sample_prompt = tokenizer.decode(batch_inputs['input_ids'][0])
            print(f"Prompt: {sample_prompt}")
            print(f"Correct answer: {batch_answers[0]}")
        
        # Generate predictions using greedy decoding
        with torch.no_grad():
            generated_outputs = model.generate(
                batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,
                max_new_tokens=10,  # Generate more tokens to capture full answer
                do_sample=False,   # Use greedy decoding
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Process each example in the batch
        for j, (output, correct_answer) in enumerate(zip(generated_outputs, batch_answers)):
            # Extract generated tokens (excluding input)
            input_length = batch_inputs.input_ids[j].size(0)
            generated_ids = output[input_length:] if input_length < len(output) else []
            
            # Decode the generated tokens
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Get first sentence or line of generation (the answer)
            generated_answer = generated_text.split('\n')[0].split('.')[0].strip()
            
            # Normalize both for comparison
            gen_norm = generated_answer.lower().strip()
            correct_norm = correct_answer.lower().strip()
            scores = scorer.score(correct_norm, gen_norm)
            
            # Check for match or partial match (answer might be cut off)
            is_correct = (gen_norm == correct_norm or 
                         gen_norm.startswith(correct_norm) or 
                         correct_norm.startswith(gen_norm) or
                         scores['rouge1'].fmeasure > 0.9)
            
            if is_correct:
                correct += 1
            
            # Debug: print prediction details
            if debug:
                print(f"\n=== DEBUG: Direct Answer Prediction for example {i*batch_size + j} ===")
                print(f"Prompt ending: ...{batch_prompts[j][-50:]}")  # Last part of prompt
                print(f"Generated text: '{generated_text}'")
                print(f"Extracted answer: '{generated_answer}'")
                print(f"Normalized generated: '{gen_norm}'")
                print(f"Correct answer: '{correct_answer}'")
                print(f"Normalized correct: '{correct_norm}'")
                print(f"Correct? {'✓' if is_correct else '✗'}")
                print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
                print(f"")
                print("-" * 50)
    
    accuracy = correct / total
    return accuracy

def evaluate_qa_by_type(model, tokenizer, test_dataset, args):
    """Evaluate QA accuracy by question type with batched processing."""
    # Results dictionaries for both formats
    option_results = {}
    direct_results = {}
    
    # Collect prompts and answers for each QA type, for both formats
    qa_type_option_prompts = {q_field: [] for q_field, _ in QA_FIELDS}
    qa_type_direct_prompts = {q_field: [] for q_field, _ in QA_FIELDS}
    
    # Debug: print an example record from test dataset
    if args.debug and len(test_dataset) > 0:
        print("\n=== DEBUG: Sample data record ===")
        sample_idx = 0
        sample = test_dataset[sample_idx]
        print(f"Record index: {sample_idx}")
        for field, value in sample.items():
            if field.endswith("_q") or field.endswith("_a"):
                print(f"{field}: {value}")
    
    # Determine which evaluation formats to use
    eval_option = args.eval_format in ["option", "both"]
    eval_direct = args.eval_format in ["direct", "both"]
    
    # Generate prompts for both evaluation formats
    for i in range(min(args.num_samples, len(test_dataset))):
        for q_field, a_field in QA_FIELDS:
            question = test_dataset[i][q_field]
            correct_answer = test_dataset[i][a_field]
            
            # Create prompts for both formats if needed
            if eval_option:
                option_prompt, option_correct = create_multiple_choice_prompt(
                    question, correct_answer, test_dataset, a_field
                )
                if option_prompt and option_correct:
                    qa_type_option_prompts[q_field].append((option_prompt, option_correct))
            
            if eval_direct:
                direct_prompt, direct_correct = create_direct_answer_prompt(
                    question, correct_answer, test_dataset, a_field
                )
                if direct_prompt and direct_correct:
                    qa_type_direct_prompts[q_field].append((direct_prompt, direct_correct))
    
    # Debug: print count of examples per question type
    if args.debug:
        print("\n=== DEBUG: Number of examples per question type ===")
        for q_field in qa_type_option_prompts:
            if eval_option:
                print(f"{q_field} (option format): {len(qa_type_option_prompts[q_field])} examples")
            if eval_direct:
                print(f"{q_field} (direct format): {len(qa_type_direct_prompts[q_field])} examples")
    
    # Evaluate option-based format
    if eval_option:
        print("\n==== Evaluating OPTION-BASED Format ====")
        option_total_correct = 0
        option_total_samples = 0
        
        for q_field, prompts_and_answers in qa_type_option_prompts.items():
            print(f"Evaluating {q_field} questions (option-based)...")
            
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
            option_results[q_field] = accuracy
            print(f"{q_field} accuracy: {accuracy:.4f}")
            
            # Accumulate totals for overall accuracy
            option_total_correct += accuracy * len(prompts_and_answers)
            option_total_samples += len(prompts_and_answers)
        
        # Calculate overall accuracy from accumulated totals
        option_results["overall"] = option_total_correct / option_total_samples if option_total_samples > 0 else 0
    
    # Evaluate direct answer format
    if eval_direct:
        print("\n==== Evaluating DIRECT ANSWER Format ====")
        direct_total_correct = 0
        direct_total_samples = 0
        
        for q_field, prompts_and_answers in qa_type_direct_prompts.items():
            print(f"Evaluating {q_field} questions (direct answer)...")
            
            # Print debug info for the first QA type only if in debug mode
            qa_debug = args.debug and q_field == QA_FIELDS[0][0]
            
            accuracy = score_direct_answers(
                model, 
                tokenizer, 
                prompts_and_answers, 
                args.device, 
                batch_size=args.batch_size,
                debug=qa_debug
            )
            direct_results[q_field] = accuracy
            print(f"{q_field} accuracy: {accuracy:.4f}")
            
            # Accumulate totals for overall accuracy
            direct_total_correct += accuracy * len(prompts_and_answers)
            direct_total_samples += len(prompts_and_answers)
        
        # Calculate overall accuracy from accumulated totals
        direct_results["overall"] = direct_total_correct / direct_total_samples if direct_total_samples > 0 else 0
    
    # Combine results for return
    combined_results = {
        "option": option_results,
        "direct": direct_results
    }
    
    return combined_results

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
    
    # Collect all possible answers by category/field for proper MCQ generation
    print("Collecting all possible answers by category...")
    all_answers_by_field = {}
    for _, a_field in QA_FIELDS:
        # Get unique answers for this field
        unique_answers = set()
        for i in range(len(test_dataset)):
            answer = test_dataset[i][a_field]
            unique_answers.add(answer)
        
        # Store as list
        all_answers_by_field[a_field] = list(unique_answers)
        print(f"Collected {len(unique_answers)} unique answers for {a_field}")
    
    # Make this available for MCQ generation
    create_multiple_choice_prompt.all_answers_by_field = all_answers_by_field
    create_direct_answer_prompt.all_answers_by_field = all_answers_by_field
    
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
    print(f"Evaluation format: {args.eval_format}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Evaluate the model
    print(f"Evaluating model on {args.num_samples} samples...")
    combined_results = evaluate_qa_by_type(model, tokenizer, test_dataset, args)
    
    # Print overall results
    if "option" in combined_results and combined_results["option"]:
        print("\n===== OPTION-BASED FORMAT RESULTS =====")
        for question_type, accuracy in combined_results["option"].items():
            print(f"{question_type}: {accuracy:.4f}")
        print(f"\nOption-Based Overall Accuracy: {combined_results['option']['overall']:.4f}")
    
    if "direct" in combined_results and combined_results["direct"]:
        print("\n===== DIRECT ANSWER FORMAT RESULTS =====")
        for question_type, accuracy in combined_results["direct"].items():
            print(f"{question_type}: {accuracy:.4f}")
        print(f"\nDirect Answer Overall Accuracy: {combined_results['direct']['overall']:.4f}")

if __name__ == "__main__":
    main()