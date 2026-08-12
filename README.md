# Recipe Generation from Ingredients

This repository contains the training, generation, and evaluation notebooks for the 3 autoregressive language models that writes recipes from a list of ingredients.
Built for ECE 1508 (Deep Generative Models, Summer 2026), the project compares fine-tuning a pretrained checkpoint against training from scratch under a limited compute budget.

## Project Overview

Each recipe is modelled as a single token sequence holding the ingredients, the title, and the directions, and the joint distribution is learned by maximum likelihood on a 250,000 recipe sample from RecipeNLG.
We give a model a set of ingredients and it outputs a title and a set of directions. The 3 models were trained on the same data, each given 6 to 7 hours on a Google Colab T4.

| Label             | Checkpoint directory              | Parameters | Epochs | Learning rate |
|-------------------|-----------------------------------|------------|--------|---------------|
| Pretrained-FT     | `model_pretrained_250k_ingfirst`  | 124M       | 2      | 5e-5          |
| Scratch-GPT2      | `model_250k_ingfirst`             | 45M        | 5      | 1e-3          |
| Scratch-Minimal   | `model_minimal_250k_ingfirst`     | 71M        | 5      | 1e-3          |

## Repository Structure

All the notebooks are grouped by stage, one per model in each folder.
Throughout, `{model}` is `gpt2_pretrained`, `gpt2_scratch`, or `minimal`.

- `/train`: `recipe_{model}.ipynb`. Trains a model and writes its checkpoint and logs to Drive.
- `/metrics`: `recipe_{model}__metrics.ipynb`. Reports the training run from the saved logs.
- `/generate`: `recipe_{model}__generate.ipynb`. Writes sample recipes from an ingredient list.
- `/corruption_metrics`: `recipe_{model}__corruption_metrics.ipynb`. Measures ingredient sensitivity.

Checkpoints, preprocessed cache, and the training logs are under our google drive: `/content/drive/MyDrive/recipe_gpt2/`.

## Notebooks

**train**
Summary: Serializes the recipes, tokenizes, and trains by maximum likelihood with causal masking. Checkpoints roughly once an hour so a disconnected run can be resumed.
Input: The RecipeNLG CSV, or the preprocessed cache if one already exists.
Output: A checkpoint directory holding the model, the tokenizer, `training_history.csv`, and `training_summary.json`.

**metrics**
Summary: Plots the saved training curves. No model is loaded.
Input: `training_history.csv` and `training_summary.json` from a checkpoint directory.  
Output: A printed run summary, plus loss and perplexity plots.

**generate**
Summary: Loads a checkpoint and writes recipes from an ingredient list.
Input: A checkpoint directory, and a comma separated ingredient string. We have a predefined 5 prompts set up.
Output: A printed title and directions per prompt. Sampling is stochastic, so reruns differ.

**corruption_metrics**
Summary: Swaps ingredients one at a time and measures the loss on the unchanged directions, showing whether a model rewrites its directions or only swaps words.
Input: A checkpoint directory and the preprocessed cache, scored on 2,500 held-out recipes.
Output: A printed table of mean loss delta at 6 different corruption levels, the ceiling as a share of baseline loss, and a plot with error bars.

## Running the Notebooks

First need to mount the Drive, set `MODEL_DIR` to the checkpoint being evaluated, and run the cells in order.
The corruption notebooks selects a model by uncommenting one of three `MODEL_DIR` lines at the top instead.
Corruption runs take a while, since each recipe is scored six times.

## Results

Generation metrics on 50 held-out samples, with the corruption ceiling.

|                 | PPL  | ROUGE-1 | ROUGE-L | BLEU-4 | Coverage | Ceiling         |
|-----------------|------|---------|---------|--------|----------|-----------------|
| Scratch-GPT2    | 5.09 | 0.368   | 0.255   | 0.060  | 64.97%   | +0.4664 (30.4%) |
| Scratch-Minimal | 5.57 | 0.367   | 0.244   | 0.065  | 60.29%   | +0.4210 (25.6%) |
| Pretrained-FT   | 5.91 | 0.321   | 0.217   | 0.049  | 59.31%   | +0.3279 (18.2%) |

The two from-scratch models are close on text overlap and separate on coverage. 
Scratch-GPT2 also shows the steepest corruption curve, meaning it is the most sensitive to individual ingredients. 
Pretrained-FT ranks last on every measure, but its loss did not fully converge when the time budget ran out.

## Requirements

Google Colab with a T4 GPU and Google Drive.
RecipeNLG dataset.
PyTorch, Hugging Face Transformers, and Matplotlib.
