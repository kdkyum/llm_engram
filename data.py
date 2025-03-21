import torch
import random
from evaluate_qa import create_multiple_choice_prompt, QA_FIELDS


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
        
        # Convert input_ids to tensors to catch any issues early
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Create labels for causal LM (same as input_ids)
        inputs["labels"] = input_ids.copy()
        
        return inputs
