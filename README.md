# LLM Engram: Evaluating Fact Memorization in Language Models

This repository contains code for fine-tuning various language models on synthetic personal biography datasets (bioS) and evaluating their performance on question-answering tasks to assess their ability to encode and recall factual information.

## Supported Models

- GPT-2 (small, medium, large, xl)
- GPT-J (6B)
- GPT-NeoX (20B)
- Llama 3.2 (1B, 3B)
- Llama 3.1 (8B)

## Setup

Make sure you have the required dependencies installed:

```bash
pip install transformers datasets torch accelerate tqdm numpy wandb peft
```

## Dataset

This project uses the `minsungkim/bioS_v1` dataset available on Hugging Face's dataset hub. The dataset contains synthetic personal biographies and related question-answer pairs. The data can be loaded using:

```python
from datasets import load_dataset
ds = load_dataset("minsungkim/bioS_v1")
```

Each entry in the dataset contains:
- Personal information (name, birth date, birth city, university, etc.)
- Biography text in different formats (bioS, bioS_fullname, bioS_multi5_permutes)
- QA pairs for six types of information (birth date, birth city, university, major, employer, company city)

## Training

### Fine-tuning a Single Model

You can fine-tune any of the supported language models using the `train.py` script:

#### Smaller Models (GPT-2)

```bash
python train.py \
  --model_name_or_path gpt2-xl \
  --model_type gpt2 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --fp16 \
  --wandb_project llm-engram \
  --freeze_embeddings
```

#### Larger Models (GPT-J, Llama)

For larger models, it's recommended to use LoRA for parameter-efficient fine-tuning:

```bash
python train.py \
  --model_name_or_path meta-llama/Llama-3.2-3B \
  --model_type llama \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --fp16 \
  --wandb_project llm-engram \
  --lora \
  --lora_r 8 \
  --lora_alpha 16
```

### Training Multiple Models

To train multiple models in sequence, use the `train_all_models.py` script:

```bash
python train_all_models.py \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_samples 1000 \
  --bio_field bioS \
  --fp16 \
  --lora
```

Key parameters:
- `--model_name_or_path`: Model identifier from HuggingFace or local path
- `--model_type`: Type of model architecture (gpt2, gpt-j, gpt-neox, llama)
- `--num_train_epochs`: Number of training epochs
- `--per_device_train_batch_size`: Batch size per device
- `--gradient_accumulation_steps`: Gradient accumulation steps
- `--learning_rate`: Learning rate
- `--fp16`: Use mixed precision training
- `--bio_field`: Which bio field to use for training (bioS, bioS_fullname, or bioS_multi5_permutes)
- `--wandb_project`: Project name for Weights & Biases logging
- `--freeze_embeddings`: Freeze model embeddings
- `--freeze_layers`: Comma separated layer indices to freeze
- `--lora`: Use LoRA for parameter-efficient fine-tuning
- `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA specific parameters

## Evaluation

### Evaluating a Single Model

For a detailed evaluation of a fine-tuned model:

```bash
python evaluate_qa.py \
  --model_path ./model-output/gpt2-xl-bioS \
  --model_type gpt2 \
  --num_samples 500 \
  --batch_size 8 \
  --fp16
```

### Evaluating Multiple Models

To evaluate and compare multiple models at once:

```bash
python run_multi_model_eval.py \
  --num_samples 500 \
  --batch_size 8 \
  --fp16 \
  --results_dir ./results
```

Key parameters:
- `--model_path`: Path to the fine-tuned model or HF model name
- `--model_type`: Type of model (gpt2, gpt-j, gpt-neox, llama)
- `--num_samples`: Number of examples to evaluate
- `--batch_size`: Batch size for evaluation
- `--fp16`: Use mixed precision for evaluation
- `--seed`: Random seed for reproducibility
- `--device`: Device to use (cuda/cpu)

The evaluation script:
1. Loads the test data
2. For each QA type, creates multiple-choice questions (1 correct + 3 incorrect options)
3. Computes accuracy for each QA type and overall
4. Presents results by question type

## QA Measurement Approach

The QA evaluation uses a multiple-choice format:
1. For each QA field (e.g., birth date, university, employer), we create a question
2. We present 4 possible answers (A, B, C, D), with only one being correct
3. Incorrect options are randomly sampled from other records in the dataset
4. The model selects the option with the highest probability
5. Accuracy is calculated for each QA type and overall

For a more thorough standalone evaluation after training:

```bash
python evaluate_qa.py --model_path ./gpt2-xl-bios/best_model_qa_0.xxxx --num_samples 1000
```