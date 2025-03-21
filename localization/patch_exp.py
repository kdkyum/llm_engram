from ast import literal_eval
import functools
import json
import os
import random
import shutil
import argparse

# Scienfitic packages
import numpy as np
import pandas as pd
import torch
import datasets
from torch import cuda
torch.set_grad_enabled(False)

# Visuals
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(context="notebook",
        rc={"font.size":16,
            "axes.titlesize":16,
            "axes.labelsize":16,
            "xtick.labelsize": 16.0,
            "ytick.labelsize": 16.0,
            "legend.fontsize": 16.0})
palette_ = sns.color_palette("Set1")
palette = palette_[2:5] + palette_[7:]
sns.set_theme(style='whitegrid')

# Utilities
from localization.general_utils import (
  ModelAndTokenizer,
  make_inputs,
)

from localization.patchscopes_utils import *
from rouge_score import rouge_scorer
from tqdm import tqdm
tqdm.pandas()

from datasets import load_dataset

model_to_hook = {
    "EleutherAI/pythia-6.9b": set_hs_patch_hooks_neox,
    "EleutherAI/pythia-12b": set_hs_patch_hooks_neox,
    "meta-llama/Llama-2-13b-hf": set_hs_patch_hooks_llama,
    "lmsys/vicuna-7b-v1.5": set_hs_patch_hooks_llama,
    "./stable-vicuna-13b": set_hs_patch_hooks_llama,
    "CarperAI/stable-vicuna-13b-delta": set_hs_patch_hooks_llama,
    "EleutherAI/gpt-j-6B": set_hs_patch_hooks_gptj
}

def format_prompt(example, question_key):
    """Format prompts for the experiment."""
    question = example[question_key]
    prompt = f"Q: Where was Cameron Diaz born?\nAnswer: San Diego, California.\n\nQ: {question}\nAnswer:"
    return prompt

def get_subject_token_position(mt, prompt, subject_name):
    """
    Get the start and end token positions for a subject name in a tokenized prompt.
    
    Args:
        mt: ModelAndTokenizer object
        prompt: The text prompt
        subject_name: The subject name to find in the prompt
    
    Returns:
        tuple: (start_token_position, end_token_position) - inclusive range
    """
    # Tokenize the prompt and the subject name
    prompt_tokens = mt.tokenizer.encode(prompt)
    subject_tokens = mt.tokenizer.encode(subject_name, add_special_tokens=False)

    for i in range(len(prompt_tokens) - len(subject_tokens) + 1):
        if prompt_tokens[i:i+len(subject_tokens)] == subject_tokens:
            start_idx = i
            end_idx = i + len(subject_tokens)
            
            return start_idx, end_idx
    
    # If we get here, we couldn't find an exact token match
    print(f"Could not find exact token sequence for '{subject_name}' in the prompt")
    
    # As a fallback, try to find the approximate position
    char_loc = prompt.index(subject_name)
    tokens_before = mt.tokenizer.encode(prompt[:char_loc])
    approximate_start = len(tokens_before) - 1  # Adjust for the leading special token
    
    subject_with_context = prompt[char_loc:char_loc+len(subject_name)+10]  # Add some context
    tokens_subject_with_context = mt.tokenizer.encode(subject_with_context)
    approximate_end = approximate_start + len(tokens_subject_with_context) - 2  # Adjust for special tokens
    
    print(f"Falling back to approximate token positions: {approximate_start} to {approximate_end}")
    return approximate_start, approximate_end


def group_names_by_token_length(mt, train_dataset):
    """Group names in the dataset by their token length."""
    name_to_token_length = {}
    for example in train_dataset:
        name = example['name']
        tokens = mt.tokenizer.encode(name, add_special_tokens=False)
        token_length = len(tokens)
        if token_length not in name_to_token_length:
            name_to_token_length[token_length] = []
        name_to_token_length[token_length].append(name)
    return name_to_token_length

def check_model_correctness(mt, prompt, expected_answer, rouge_threshold=0.5):
    """Check if the model produces a correct answer for the given prompt.
    
    Args:
        mt: ModelAndTokenizer object
        prompt: Input prompt
        expected_answer: Expected answer
        rouge_threshold: Threshold for ROUGE-1 score to consider the answer correct
        
    Returns:
        tuple: (is_correct, rouge_score, model_answer)
    """
    inp_target = make_inputs(mt.tokenizer, [prompt], mt.device)
    max_gen_len = 15
    output_toks = mt.model.generate(
            inp_target["input_ids"],
            max_length=len(inp_target["input_ids"][0]) + max_gen_len,
            pad_token_id=mt.model.generation_config.eos_token_id,
        )[0][len(inp_target["input_ids"][0]) :]

    result = mt.tokenizer.decode(output_toks)
    
    # Extract first line of the result
    model_answer = result.split('\n')[0].strip() if '\n' in result else result.strip()
    
    # Calculate ROUGE score
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    score = scorer.score(expected_answer, model_answer)
    rouge1_score = score['rouge1'].fmeasure
    
    # Check if the answer is correct based on the threshold
    is_correct = rouge1_score >= rouge_threshold
    
    return is_correct, rouge1_score, model_answer

def run_patching_experiment(mt, q_key, train_dataset, src_idx, output_dir, only_correct=False, rouge_threshold=0.5):
    """Run patching experiment for a specific question type."""
    a_key = q_key.replace('_q', '_a')
    print(f"\n{'='*80}\nProcessing question category: {q_key} / {a_key}\n{'='*80}")

    source_subject_name = train_dataset[src_idx]["name"]
    name_to_token_length = group_names_by_token_length(mt, train_dataset)
    # Get token length of current source subject
    source_name_tokens = mt.tokenizer.encode(source_subject_name, add_special_tokens=False)
    source_name_token_length = len(source_name_tokens)
    print(f"Source name '{source_subject_name}' has token length: {source_name_token_length}")

    # Find names with same token length (excluding the source name)
    same_length_names = [n for n in name_to_token_length.get(source_name_token_length, []) 
                         if n != source_subject_name]

    if not same_length_names:
        print("No other names with the same token length found.")
        # Use original target if no matching length name is found
        patching_target_name = train_dataset[1]["name"]  # Just use the next example as target
    else:
        # Randomly select a target name with the same token length
        patching_target_name = random.choice(same_length_names)

    # Find this name in the dataset to get the corresponding prompt
    target_examples = [i for i, ex in enumerate(train_dataset) if ex['name'] == patching_target_name]

    patching_target_idx = target_examples[0]
    patching_target_prompt = format_prompt(train_dataset[patching_target_idx], q_key)
    patching_expected_answer = train_dataset[src_idx][a_key]
    # if target_examples:
    #     patching_target_idx = target_examples[0]
    #     patching_target_prompt = format_prompt(train_dataset[patching_target_idx], q_key)
    #     patching_expected_answer = train_dataset[src_idx][a_key]
    # else:
    #     # Create a dummy prompt/answer
    #     prompts = []
    #     answers = []
    #     for i in range(len(train_dataset)):
    #         prompt = format_prompt(train_dataset[i], q_key)
    #         answer = train_dataset[i][a_key]
    #         prompts.append(prompt)
    #         answers.append(answer)
        
    #     source_prompt_idx = 0
    #     patching_target_prompt = prompts[source_prompt_idx + 6]  # +6 for the target subject
    #     patching_expected_answer = answers[source_prompt_idx]

    # Setup ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Define layers to test
    layers_to_test = np.arange(1, 40)  # 39 is the last layer for Llama-2-13b
    layers_to_test_reversed = layers_to_test[::-1]

    # Run patching experiments for different positions and layers
    source_prompt = format_prompt(train_dataset[src_idx], q_key)
    source_tokens = mt.tokenizer.encode(source_prompt)

    print(source_prompt)
    print(patching_target_prompt)
    
    # If only_correct is True, check if the model gives a correct answer for this prompt
    if only_correct:
        expected_answer = train_dataset[src_idx][a_key]
        is_correct, rouge1_score, model_answer = check_model_correctness(
            mt, source_prompt, expected_answer, rouge_threshold
        )
        
        if not is_correct:
            print(f"Model answer for '{source_subject_name}' is incorrect (ROUGE-1: {rouge1_score:.2f})")
            print(f"Expected: '{expected_answer}'")
            print(f"Got: '{model_answer}'")
            # Return None to indicate this example should be skipped
            return None
        else:
            print(f"Model answer for '{source_subject_name}' is correct (ROUGE-1: {rouge1_score:.2f})")
            print(f"Expected: '{expected_answer}'")
            print(f"Got: '{model_answer}'")
    
    # Use get_subject_token_position to find the token positions
    subj_first_pos, subj_last_pos = get_subject_token_position(mt, source_prompt, source_subject_name)
    
    # Fallback if positions are None
    if subj_first_pos is None or subj_last_pos is None:
        print(f"Using fallback positions for subject '{source_subject_name}'")
        subj_first_pos = 10  # An arbitrary position
        subj_last_pos = 15   # An arbitrary position
    
    print(f"Subject name '{source_subject_name}' found at token positions {subj_first_pos} to {subj_last_pos}")
    # Initialize results matrix for heatmap
    positions_range = list(range(subj_first_pos, len(source_tokens)))
    rouge_scores_matrix = np.zeros((len(layers_to_test), len(positions_range)))
    
    # Decode tokens for x-axis labels
    token_labels = []
    patching_target_prompt_tokens = mt.tokenizer.encode(patching_target_prompt)
    for position in positions_range:
        token_id = patching_target_prompt_tokens[position]
        token_text = mt.tokenizer.decode([token_id])
        token_labels.append(f"{position}: '{token_text}'")

    # Run experiments
    for layer_idx, layer in enumerate(tqdm(layers_to_test, desc=f"Processing layers for {q_key}")):
        for pos_idx, position in enumerate(tqdm(positions_range, desc=f"Processing positions for layer {layer}", leave=False)):
            result = inspect(
                mt=mt,
                prompt_source=source_prompt,
                prompt_target=patching_target_prompt,
                layer_source=layer,
                layer_target=layer,
                position_source=position,
                position_target=position,
                module="hs",
                generation_mode=True,
                max_gen_len=15
            )
            
            # Extract date from result
            extracted_result = result.split('\n')[0].strip() if '\n' in result else result.strip()
            
            # Calculate ROUGE score
            score = scorer.score(patching_expected_answer, extracted_result)
            rouge_scores_matrix[layer_idx, pos_idx] = score['rouge1'].fmeasure

    # Plot heatmap for this category and save to file
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(
        rouge_scores_matrix[::-1],  # Reverse the order of layers
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu",
        xticklabels=token_labels,
        yticklabels=[str(l) for l in layers_to_test_reversed]  # Reversed layers
    )
    plt.xlabel('Token Position and Text')
    plt.ylabel('Layer Index')
    plt.title(f'ROUGE-1 Scores for {q_key}: Different Layer-Position Combinations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(output_dir, f"heatmap_{q_key.replace('_q', '')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {output_file}")

    # Print best combinations
    flat_idx = np.argmax(rouge_scores_matrix)
    best_layer_idx = flat_idx // len(positions_range)
    best_pos_idx = flat_idx % len(positions_range)
    best_layer = layers_to_test[best_layer_idx]
    best_position = positions_range[best_pos_idx]
    best_score = rouge_scores_matrix[best_layer_idx, best_pos_idx]
    best_token = token_labels[best_pos_idx]

    print(f"Best combination for {q_key}: Layer {best_layer}, Position {best_position} ({best_token}), ROUGE-1 Score: {best_score:.4f}")
    
    return {
        'rouge_scores_matrix': rouge_scores_matrix,
        'positions_range': positions_range,
        'expected_answer': patching_expected_answer,
        'token_labels': token_labels,
        'subject_range': (subj_first_pos, subj_last_pos)  # Store subject token range
    }

def aggregate_scores(all_subject_results, question_keys, layers_to_test, output_dir):
    """Aggregate ROUGE-1 scores across subjects and token positions.
    
    Args:
        all_subject_results: Dictionary with results for each subject and question
        question_keys: List of question keys to process
        layers_to_test: Array of layer indices
        output_dir: Directory to save output plots
    """
    layers_to_test_reversed = layers_to_test[::-1]
    
    for q_key in question_keys:
        print(f"\n{'='*80}\nAggregating results for {q_key}\n{'='*80}")
        
        # Collect data from all subjects for this question
        subject_data = [data[q_key] for data in all_subject_results]
        
        # Identify token positions across subjects
        all_subject_positions = []
        all_subject_ranges = []
        
        for data in subject_data:
            subj_range = data['subject_range']  # (first_pos, last_pos)
            all_subject_ranges.append(subj_range)
            all_subject_positions.append(data['positions_range'])
        
        # Build a mapping of positions to subject indices and whether they're subject tokens
        position_map = {}
        for subj_idx, (positions, subj_range) in enumerate(zip(all_subject_positions, all_subject_ranges)):
            for pos in positions:
                if pos not in position_map:
                    position_map[pos] = []
                
                # Check if this position is within subject range
                is_subject = subj_range[0] <= pos < subj_range[1]
                position_map[pos].append((subj_idx, is_subject))
        
        # Initialize matrices for aggregated scores
        num_layers = len(layers_to_test)
        all_positions = sorted(position_map.keys())
        
        # Matrix for averaging across subjects only (for non-subject tokens)
        avg_across_subjects = np.zeros((num_layers, len(all_positions)))
        # Matrix for averaging across subjects and middle tokens
        avg_across_subjects_and_positions = np.zeros((num_layers, 1))
        
        # Counter matrices to track number of samples for averaging
        subject_counts = np.zeros((num_layers, len(all_positions)))
        subject_position_counts = np.zeros((num_layers, 1))
        
        # Collect scores from all subjects
        for pos_idx, pos in enumerate(all_positions):
            subject_pos_entries = position_map[pos]
            
            for subj_idx, is_subject in subject_pos_entries:
                # Get the original matrix and find corresponding position
                orig_matrix = subject_data[subj_idx]['rouge_scores_matrix']
                orig_positions = subject_data[subj_idx]['positions_range']
                
                if pos in orig_positions:
                    orig_pos_idx = orig_positions.index(pos)
                    
                    # For each layer
                    for layer_idx in range(num_layers):
                        score = orig_matrix[layer_idx, orig_pos_idx]
                        
                        # Add to appropriate matrices based on whether it's a subject token
                        if is_subject:
                            # For subject tokens, aggregate across both subjects and positions
                            avg_across_subjects_and_positions[layer_idx, 0] += score
                            subject_position_counts[layer_idx, 0] += 1
                        else:
                            # For non-subject tokens, aggregate only across subjects
                            avg_across_subjects[layer_idx, pos_idx] += score
                            subject_counts[layer_idx, pos_idx] += 1
        
        # Compute averages
        avg_across_subjects = np.divide(
            avg_across_subjects, 
            subject_counts, 
            out=np.zeros_like(avg_across_subjects), 
            where=subject_counts != 0
        )
        
        avg_across_subjects_and_positions = np.divide(
            avg_across_subjects_and_positions,
            subject_position_counts,
            out=np.zeros_like(avg_across_subjects_and_positions),
            where=subject_position_counts != 0
        )
        
        # Create labels for the x-axis
        pos_labels = [f"{pos}" for pos in all_positions]
        
        # Plot heatmap for average across subjects (non-subject tokens)
        plt.figure(figsize=(14, 8))
        ax = sns.heatmap(
            avg_across_subjects[::-1],  # Reverse layers for better visualization
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=pos_labels,
            yticklabels=[str(l) for l in layers_to_test_reversed]
        )
        plt.xlabel('Token Position')
        plt.ylabel('Layer Index')
        plt.title(f'Average ROUGE-1 Scores Across Subjects for {q_key}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(output_dir, f"avg_subjects_{q_key.replace('_q', '')}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved average heatmap across subjects to {output_file}")
        
        # Plot heatmap for average across subjects and positions (subject tokens)
        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(
            avg_across_subjects_and_positions[::-1],
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=["Subject Tokens"],
            yticklabels=[str(l) for l in layers_to_test_reversed]
        )
        plt.xlabel('Token Type')
        plt.ylabel('Layer Index')
        plt.title(f'Average ROUGE-1 Scores Across Subjects & Positions for {q_key}')
        plt.tight_layout()
        
        # Save the figure
        output_file = os.path.join(output_dir, f"avg_subjects_positions_{q_key.replace('_q', '')}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved average heatmap across subjects and positions to {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run patch experiments with LLMs')
    parser.add_argument('--model_path', type=str, default="/home/kdkyum/workdir/llm_engram/model-output/Llama-2-13b-hf-bioS_multi5_permutes-lr5.0e-5/best_model",
                        help='Path to the local model')
    parser.add_argument('--output_dir', type=str, default="./plots",
                        help='Directory to save output plots')
    parser.add_argument('--average', action='store_true',
                        help='Calculate and display average scores across subjects and token positions')
    parser.add_argument('--num_subjects', type=int, default=20,
                        help='Number of subjects to use for averaging (if --average is specified)')
    parser.add_argument('--only_correct', action='store_true',
                        help='Only include examples where the model produces correct answers (ROUGE-1 > 0.5)')
    parser.add_argument('--rouge_threshold', type=float, default=0.5,
                        help='ROUGE-1 score threshold for considering an answer correct')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model_name = "meta-llama/Llama-2-13b-hf"
    local_model_path = args.model_path
    
    mt = ModelAndTokenizer(
        "meta-llama/Llama-2-13b-hf",
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
        local_model_path=local_model_path
    )
    mt.set_hs_patch_hooks = model_to_hook[model_name]
    mt.model.eval()

    # Load dataset
    dataset = load_dataset("minsungkim/bioS_v1")
    train_dataset = dataset["train"]
    train_dataset = train_dataset.select(range(20))

    # Define question types to use
    question_keys = [key for key in train_dataset[0].keys() if key.endswith('_q')]
    
    if args.average:
        # Run experiments for multiple subjects and calculate averages
        num_subjects = min(args.num_subjects, len(train_dataset))
        print(f"Running experiments for {num_subjects} subjects and calculating averages")
        
        if args.only_correct:
            print(f"Only including examples where model produces correct answers (ROUGE-1 > {args.rouge_threshold})")
        
        # Store results for each subject
        all_subject_results = []
        
        # Define layers to test once (same for all experiments)
        layers_to_test = np.arange(1, 40)  # 39 is the last layer for Llama-2-13b
        
        # Run experiments for each subject
        for src_idx in range(num_subjects):
            subject_results = {}
            subject_name = train_dataset[src_idx]['name']
            print(f"\n{'='*80}\nProcessing subject {src_idx}: {subject_name}\n{'='*80}")
            
            # Flag to track if this subject has at least one correct answer
            has_correct_answer = False
            os.makedirs(os.path.join(args.output_dir, f"subject_{src_idx}"), exist_ok=True)
            
            # Run experiments for all question types for this subject
            for q_key in question_keys:
                result = run_patching_experiment(
                    mt=mt,
                    q_key=q_key,
                    train_dataset=train_dataset,
                    src_idx=src_idx,
                    output_dir=os.path.join(args.output_dir, f"subject_{src_idx}"),
                    only_correct=args.only_correct,
                    rouge_threshold=args.rouge_threshold
                )
                
                if result is not None:
                    subject_results[q_key] = result
                    has_correct_answer = True
                elif not args.only_correct:
                    # If not filtering for correct answers, include all results
                    subject_results[q_key] = result
            
            # Only add this subject's results if it has at least one correct answer
            # or if we're not filtering for correct answers
            if has_correct_answer or not args.only_correct:
                if len(subject_results) > 0:  # Only add if there are valid results
                    all_subject_results.append(subject_results)
            else:
                print(f"Subject {src_idx}: {subject_name} has no correct answers, skipping")
        
        # Only proceed with aggregation if we have results
        if all_subject_results:
            print(f"Aggregating results from {len(all_subject_results)} subjects")
            # Calculate and plot the averages
            aggregate_scores(all_subject_results, question_keys, layers_to_test, args.output_dir)
        else:
            print("No subjects with correct answers found. Cannot calculate averages.")
    else:
        # Original behavior: run experiments for a single subject
        src_idx = 0
        
        # Store results for each question category
        all_results = {}

        # Loop through all question categories
        for q_idx, q_key in enumerate(question_keys):
            all_results[q_key] = run_patching_experiment(
                mt=mt,
                q_key=q_key,
                train_dataset=train_dataset,
                src_idx=src_idx,
                output_dir=args.output_dir
            )

        # Save all results to a numpy file
        results_file = os.path.join(args.output_dir, "rouge_scores_results.npz")
        np.savez(results_file, **{k: v['rouge_scores_matrix'] for k, v in all_results.items()})
        print(f"Saved results to {results_file}")

if __name__ == "__main__":
    main()