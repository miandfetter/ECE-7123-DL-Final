# ECE-7123 Deep Learning Final Project
## Pixels to Predictions: Visual Question Answering with Fine-tuned SmolVLM

**Competition:** [Pixels to Predictions (Kaggle)](https://www.kaggle.com/c/pixels-to-predictions)

> To view rendered notebook output with plots, use NBViewer:
> [Data_Exploration.ipynb on NBViewer](https://nbviewer.org/github/miandfetter/ECE-7123-DL-Final/blob/main/Data_Exploration.ipynb?flush_cache=true)

## Repository Structure

```
ECE-7123-DL-Final/
│
├── Data_Exploration.ipynb
│   Exploratory data analysis: subject/grade distributions, answer balance,
│   text length histograms, class imbalance analysis, and weighted sampling motivation.
│   Run on Google Colab. Use NBViewer link above for rendered plots.
│
└── Training_Inference_Code/
    │
    ├── finetuning-vqa-model-bestRun.ipynb
    │   Best Kaggle notebook run. Trains on 1000 samples/epoch for 3 epochs using
    │   LoRA r=8 targeting all attention + MLP projection layers.Generates and saves a submission.csv for Kaggle.
    │
    └── colab_version_kaggle (7).ipynb
        Best Google Colab notebook run and the version submitted with model weights. Trains on 1000 samples/epoch for 2 epochs
        using LoRA r=4 targeting attention layers only, with a lower LR and linear
        warmup scheduler. 
```
---
---

## How to Reproduce

### Prerequisites

```bash
pip install peft==0.18.1 bitsandbytes accelerate datasets pillow transformers pandas
```

### Kaggle 

1. Upload `finetuning-vqa-model-bestRun.ipynb` to a Kaggle notebook.
2. Add the `pixels-to-predictions` competition dataset.
3. Enable GPU (T4 x1) under **Settings → Accelerator**.
4. Run all cells. The best LoRA checkpoint is saved to `/kaggle/working/run11/` and a `submission.csv` is generated.

### Google Colab

1. Open `colab_version_kaggle (7).ipynb` in Google Colab.
2. Set your Kaggle API token as a Colab secret (`KAGGLE_API_TOKEN`).
3. Mount Google Drive (used to save checkpoints between sessions).
4. Enable a GPU runtime (T4 recommended).
5. Run all cells. Checkpoints save to `MyDrive/vqa-checkpoints/first_colab/`.

---




## Key Design Decisions

- **Why LoRA instead of full fine-tuning?** The 4-bit quantized base model cannot be updated directly; LoRA adapters train on top of frozen quantized weights, which is the standard approach for QLoRA.
- **Why a small training subset?** Full dataset training would take several hours per epoch on a T4. Smaller experiments were performed to see if changes made meaningful differences in performance, and favorable results were expanded to training runs with more training examples to optimize GPU training time availble. 
---

## Task Overview

This project tackles **multimodal multiple-choice question answering** over K–8 science curriculum. Each sample consists of:

- An image (diagram, chart, or photograph)
- A multiple-choice question with 2–5 answer options (labeled A–E)
- Optional context: a lecture passage and/or a hint
- Subject (`natural science`, `social science`, or `language science`) and grade level

The goal is to predict the correct answer letter for each question in the test set.

---

## Dataset

| Split | Samples | Has Image |
|-------|---------|-----------|
| Train | 3,109   | 100%      |
| Val   | 1,048   | 100%      |
| Test  | 1,008   | 100%      |

**Key characteristics identified during data exploration:**
- **Subject imbalance:** ~73% natural science, ~24% social science, ~3% language science
- **Grade distribution:** Grades 3–8 account for ~90% of samples; grades 1, 10, 12 are rare
- **Context length:** Lecture + hint passages can be several hundred words, which stresses the 500M model's effective context window
- **Answer balance:** Correct answers are distributed roughly uniformly across answer slots (A–E)

---

## Approach

### Model

Fine-tuned **[SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)**, a 500M-parameter Vision Language Model from HuggingFace. It was chosen for its small footprint (fits on a free Kaggle/Colab T4 GPU) while still having strong multimodal capabilities.

### Memory Efficiency: 4-bit Quantization

The base model is loaded in **4-bit NF4 quantization** via `bitsandbytes` to reduce GPU memory usage from ~2 GB (float16) to ~512 MB, enabling training on a 16 GB T4.


### Parameter-Efficient Fine-Tuning: LoRA

**Low-Rank Adaptation (LoRA)** was used via `peft` to fine-tune only a small fraction of model parameters (~0.2–0.9% of total), keeping the quantized base model frozen.

| Run | LoRA rank (`r`) | LoRA alpha | Target modules | Trainable params |
|-----|-----------------|------------|----------------|------------------|
| Colab (not best, but submitted)  | 4 | 16 | q, k, v, o only            | ~1.0M (0.20%) |




