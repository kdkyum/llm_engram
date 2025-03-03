#!/usr/bin/env python3

import argparse
import json
import os
from datasets import load_dataset
from tqdm import tqdm

# Definition of QA field pairs from evaluate_qa.py
QA_FIELDS = [
    ("bdate_q", "bdate_a"),
    ("bcity_q", "bcity_a"),
    ("university_q", "university_a"),
    ("major_q", "major_a"),
    ("employer_q", "employer_a"),
    ("company_city_q", "company_city_a")
]

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute unique answers by category")
    parser.add_argument("--output_file", type=str, default="unique_answers.json", 
                        help="Path to save the unique answers")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Loading dataset...")
    dataset = load_dataset("minsungkim/bioS_v1")
    train_dataset = dataset["train"]
    
    print(f"Collecting unique answers for each category from {len(train_dataset)} examples...")
    
    # Dictionary to store unique answers by field
    all_answers_by_field = {}
    
    # Process each answer field
    for _, a_field in tqdm(QA_FIELDS, desc="Processing fields"):
        # Use a set for faster unique collection
        unique_answers = set()
        
        # Process dataset with progress bar
        for i in tqdm(range(len(train_dataset)), desc=f"Processing {a_field}", leave=False):
            answer = train_dataset[i][a_field]
            unique_answers.add(answer)
        
        # Convert set to list for JSON serialization
        all_answers_by_field[a_field] = list(unique_answers)
        print(f"Collected {len(unique_answers)} unique answers for {a_field}")
    
    # Save to JSON file
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(all_answers_by_field, f)
    
    print("Done!")

if __name__ == "__main__":
    main()