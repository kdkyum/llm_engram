#!/usr/bin/env python3
import torch
import random
from transformers import TrainerCallback
from evaluate_qa import create_multiple_choice_prompt, create_direct_answer_prompt, QA_FIELDS, score_answers, score_direct_answers
import wandb
import os


def get_device_fix_patch():
    """
    Monkey patch the fixed_cross_entropy function in transformers to handle multi-device scenarios
    """
    from transformers.loss import loss_utils
    
    original_fixed_cross_entropy = loss_utils.fixed_cross_entropy
    
    def patched_fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs):
        """
        Patched version that ensures tensors are on the same device before operations
        """
        import torch.nn.functional as F
        
        # Get target device from logits
        target_device = logits.device
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index, reduction=reduction)
        
        # Move num_items_in_batch to the device of loss if it's a tensor
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(target_device)

        if reduction == "sum":
            loss = loss / num_items_in_batch
        return loss
    
    # Patch the function
    loss_utils.fixed_cross_entropy = patched_fixed_cross_entropy


def freeze_model_embeddings(model, args):
    print("Freezing model embeddings and layer normalization layers...")
    
    # Freeze embeddings based on model type
    if args.model_type == "gpt-j":
        # GPT-J specific
        for param in model.transformer.wte.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        # Freeze layer norms
        for name, module in model.named_modules():
            if 'ln' in name.lower() or 'layernorm' in name.lower() or 'norm' in name.lower():
                for param in module.parameters():
                    param.requires_grad = False
                print(f"Froze layer norm: {name}")
                
    elif args.model_type == "gpt-neox":
        # GPT-NeoX specific
        for param in model.gpt_neox.embed_in.parameters():
            param.requires_grad = False
        for param in model.embed_out.parameters():
            param.requires_grad = False
        # Freeze layer norms
        for name, module in model.named_modules():
            if 'ln' in name.lower() or 'layernorm' in name.lower() or 'norm' in name.lower():
                for param in module.parameters():
                    param.requires_grad = False
                print(f"Froze layer norm: {name}")
                
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
            
        # Freeze layer norms in LLaMA models
        for name, module in model.named_modules():
            if 'norm' in name.lower() or 'ln' in name.lower() or 'layernorm' in name.lower():
                for param in module.parameters():
                    param.requires_grad = False
                print(f"Froze layer norm: {name}")
                
    elif args.model_type == "gpt2":
        for param in model.transformer.wte.parameters():
            param.requires_grad = False
        for param in model.transformer.wpe.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        # Freeze layer norms
        for name, module in model.named_modules():
            if 'ln' in name.lower() or 'layernorm' in name.lower():
                for param in module.parameters():
                    param.requires_grad = False
                print(f"Froze layer norm: {name}")
                
    elif args.model_type == "olmo":
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False
        for param in model.model.rotary_emb.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        # Freeze layer norms
        for name, module in model.named_modules():
            if 'norm' in name.lower() or 'ln' in name.lower():
                for param in module.parameters():
                    param.requires_grad = False
                print(f"Froze layer norm: {name}")


def enable_gradient_checkpointing(model, args):
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
    