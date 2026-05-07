# TinyStories SLM with Mistral Tokenizer

This project trains a small GPT-style Small Language Model (SLM) from scratch on the TinyStories dataset using the `mistralai/Mistral-7B-v0.1` tokenizer.

The main purpose of this experiment is to test whether replacing the GPT-2 tokenizer with a Mistral-style tokenizer makes a meaningful difference in a small story-generation model.

The model is trained entirely in Google Colab using local runtime storage only. Google Drive is not required.

---

## Project Goal

The goal of this project is to build a small language model that can generate short, simple stories similar to the TinyStories dataset.

This experiment focuses on:

- Using the Mistral tokenizer instead of the GPT-2 tokenizer
- Training a GPT-like transformer model from scratch
- Using a vocabulary size of approximately 32,000 tokens
- Using a context length of 2048 tokens
- Running the full experiment for free on Google Colab
- Saving and downloading checkpoints manually without Google Drive

---

## Dataset Used

This project uses the TinyStories dataset:

```python
roneneldan/TinyStories
```

TinyStories contains short, simple English stories. It is useful for experimenting with small language models because the language structure is simpler compared to large web-scale datasets.

The dataset is loaded using Hugging Face `datasets`:

```python
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")
```

---

## Tokenizer Used

This project uses:

```python
mistralai/Mistral-7B-v0.1
```

Only the tokenizer is used. The Mistral 7B model weights are not used.

The tokenizer is loaded using:

```python
from transformers import AutoTokenizer

TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
```

The expected vocabulary size is:

```text
vocab_size = 32000
```

This replaces the older GPT-2 tokenizer approach:

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
```

with the Mistral tokenizer:

```python
tokenizer.encode(text, add_special_tokens=False)
```

---

## Why Use the Mistral Tokenizer?

The experiment uses the Mistral tokenizer because:

- It has a smaller vocabulary than GPT-2's 50,257-token vocabulary
- It is modern and widely used in current open-source LLM workflows
- It supports English text well
- It is suitable for experimenting with small language models
- It allows a compact 32k vocabulary setup

The tokenizer converts text into token IDs before the model sees the data.

Example:

```python
sample_text = "Once upon a time there was a pumpkin."

ids = tokenizer.encode(sample_text, add_special_tokens=False)

print(ids)
print(tokenizer.decode(ids))
```

---

## Model Overview

The project implements a small GPT-style decoder-only transformer model.

The model includes:

- Token embedding layer
- Positional embedding layer
- Causal self-attention
- Feed-forward MLP layers
- Layer normalization
- Transformer blocks
- Final language modeling head
- Weight tying between token embeddings and output head

The model is trained using next-token prediction.

That means for a sequence like:

```text
Once upon a time
```

The model learns to predict the next token after each previous token.

---

## Model Configuration

The current model configuration is:

```python
vocab_size = tokenizer.vocab_size
block_size = 2048

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = True
```

Meaning:

| Parameter | Meaning |
|---|---|
| `vocab_size` | Number of tokens in the tokenizer vocabulary |
| `block_size` | Maximum context length used during training |
| `n_layer` | Number of transformer blocks |
| `n_head` | Number of attention heads |
| `n_embd` | Embedding dimension |
| `dropout` | Regularization to reduce overfitting |
| `bias` | Whether linear layers use bias terms |

The model is intentionally small so it can be trained on Google Colab.

---

## Context Length

This project uses:

```python
block_size = 2048
```

This means the model can see up to 2048 tokens at a time during training.

A larger context length allows the model to learn from longer sequences, but it also increases GPU memory usage.

For faster experiments, this can be reduced to:

```python
block_size = 512
```

or:

```python
block_size = 1024
```

For the final intended setup, this project keeps:

```text
context_length = 2048
```

---

## Storage Setup

This project does not use Google Drive.

All files are stored inside the local Colab runtime:

```text
/content/tinystories_mistral_slm
```

The folder structure is:

```text
/content/tinystories_mistral_slm
│
├── data
│   ├── train.bin
│   └── val.bin
│
├── checkpoints
│   ├── best_model_params.pt
│   └── tinystories_mistral_slm_full.pt
│
└── tokenizer
```

Important:

Colab local storage is temporary. Files may be deleted when the runtime disconnects or resets.

Therefore, checkpoints should be downloaded manually after training.

---

## Installation

Run the following in Google Colab:

```python
!pip -q install -U datasets transformers sentencepiece tqdm accelerate
```

Required libraries:

- `torch`
- `datasets`
- `transformers`
- `sentencepiece`
- `numpy`
- `tqdm`
- `accelerate`

---

## How the Data Processing Works

The TinyStories dataset is loaded from Hugging Face.

Each story is tokenized using the Mistral tokenizer.

An EOS token is added after each story:

```python
ids.append(eos_id)
```

This helps the model learn where one story ends and another story begins.

The tokenized data is saved as binary files:

```text
train.bin
val.bin
```

The project uses `np.uint16` for storing token IDs because the Mistral tokenizer vocabulary size is around 32,000, which fits safely within the `uint16` limit of 65,535.

```python
arr = np.memmap(
    filename,
    dtype=np.uint16,
    mode="w+",
    shape=(arr_len,)
)
```

This keeps the dataset compact and efficient to load during training.

---

## Train and Validation Split

TinyStories already provides separate dataset splits.

The project creates:

```text
train.bin
val.bin
```

These binary files are then memory-mapped using:

```python
train_data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
val_data = np.memmap(val_bin_path, dtype=np.uint16, mode="r")
```

Memory mapping allows the notebook to access large token files efficiently without loading everything into RAM at once.

---

## Batch Creation

Training batches are created using random chunks from the tokenized data.

For each batch:

- `x` contains a sequence of tokens
- `y` contains the same sequence shifted by one token

Example:

```text
x: Once upon a time
y: upon a time there
```

This teaches the model to predict the next token.

The batch function returns:

```python
X, Y = get_batch("train")
```

where:

```python
X.shape = [batch_size, block_size]
Y.shape = [batch_size, block_size]
```

---

## Training Setup

The training configuration is:

```python
learning_rate = 3e-4
max_iters = 5000
warmup_steps = 200
min_lr = 3e-5

eval_interval = 250
eval_iters = 50

batch_size = 1
gradient_accumulation_steps = 16
```

Because the context length is 2048, the batch size is kept small to fit within Colab GPU memory.

Gradient accumulation is used to simulate a larger effective batch size:

```text
effective_batch_size = batch_size × gradient_accumulation_steps
```

In this project:

```text
effective_batch_size = 1 × 16 = 16
```

---

## Mixed Precision Training

The notebook automatically checks whether the GPU supports `bfloat16`.

If supported, it uses:

```python
bfloat16
```

Otherwise, it uses:

```python
float16
```

This reduces memory usage and can speed up training on compatible GPUs.

```python
use_bf16 = device_type == "cuda" and torch.cuda.is_bf16_supported()
dtype = "bfloat16" if use_bf16 else "float16"
```

---

## Optimizer and Scheduler

The optimizer used is AdamW:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    eps=1e-9
)
```

The learning rate schedule has two phases:

1. Warmup phase
2. Cosine decay phase

This helps stabilize training in the beginning and gradually reduce the learning rate later.

---

## Loss Function

The model is trained using cross-entropy loss:

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    targets.view(-1),
    ignore_index=-1,
)
```

This is the standard loss function for next-token prediction.

Lower loss generally means the model is getting better at predicting the next token.

---

## Checkpointing

The notebook saves two types of checkpoints.

### 1. Best Model Parameters

```text
best_model_params.pt
```

This stores only the best model weights based on validation loss.

### 2. Full Checkpoint

```text
tinystories_mistral_slm_full.pt
```

This stores:

- Model weights
- Model configuration
- Tokenizer name
- Training loss history
- Validation loss history

Example:

```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "config": {
        "vocab_size": config.vocab_size,
        "block_size": config.block_size,
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "dropout": config.dropout,
        "bias": config.bias,
    },
    "tokenizer_name": TOKENIZER_NAME,
    "train_loss_list": train_loss_list,
    "validation_loss_list": validation_loss_list,
}
```

---

## Manual Checkpoint Download

Because Google Drive is not used, checkpoints must be downloaded manually:

```python
from google.colab import files

files.download(full_checkpoint_path)
files.download(best_model_params_path)
```

If this step is skipped, the trained model may be lost when the Colab runtime resets.

---

## Text Generation

After training, the model can generate text from a prompt.

Example prompts:

```python
prompts = [
    "Once upon a time there was a little girl.",
    "Tom found a small box under the bed.",
    "The dog wanted to play in the garden.",
    "Lily was scared of the dark, but"
]
```

The prompt is tokenized using the same Mistral tokenizer:

```python
context_ids = tokenizer.encode(
    prompt,
    add_special_tokens=False
)
```

The model then generates new tokens:

```python
y = best_model.generate(
    context,
    max_new_tokens=120,
    temperature=0.7,
    top_k=50
)
```

Finally, the output token IDs are decoded back into text:

```python
print(tokenizer.decode(y.squeeze().tolist()))
```

---

## Generation Parameters

### Temperature

Temperature controls randomness.

```python
temperature = 0.7
```

Lower values produce safer, more predictable text.

Higher values produce more creative but less stable text.

Recommended values:

```text
0.5 to 0.8
```

### Top-k

Top-k limits the model to choosing from the most likely tokens.

```python
top_k = 50
```

Lower values make output more focused.

Higher values allow more variety.

Recommended values:

```text
30 to 100
```

---

## Common Output Issues

During early training, the generated stories may have problems such as:

- Repeated words
- Broken grammar
- Unrelated sentences
- Characters changing randomly
- Story losing direction
- EOS token appearing as `</s>`

These issues usually happen because:

- The model is still undertrained
- The model is small
- Validation loss is still high
- Sampling is too random
- The generation function does not stop at EOS
- TinyStories contains many short stories stitched together during training

These issues do not necessarily mean the tokenizer is bad.

The tokenizer only converts text into token IDs. Story quality mainly depends on:

- Training time
- Dataset quality
- Model size
- Loss value
- Sampling settings
- Generation logic

---

## Important Note About EOS Token

The code adds an EOS token after every story:

```python
ids.append(eos_id)
```

This helps the model learn story boundaries.

However, if the generation function does not stop when EOS is produced, the output may continue into another story.

A future improvement is to stop generation when the model predicts the EOS token.

---

## Suggested Improvements

Possible improvements for better story generation:

1. Train for more steps
2. Use a lower temperature such as `0.5` or `0.6`
3. Reduce `top_k` to around `30`
4. Stop generation at the EOS token
5. Try smaller context length for TinyStories, such as `512` or `1024`
6. Increase model size if GPU memory allows
7. Track validation loss carefully
8. Compare with GPT-2 tokenizer using the same training budget

---

## Recommended Smaller Experiment

For faster experiments on Colab, use:

```python
block_size = 512
batch_size = 8
gradient_accumulation_steps = 4
```

or:

```python
block_size = 1024
batch_size = 4
gradient_accumulation_steps = 4
```

This may train faster and produce better early results for TinyStories.

For the final target setup, use:

```python
block_size = 2048
```

---

## What This Project Demonstrates

This project demonstrates how to:

- Load TinyStories from Hugging Face
- Replace GPT-2 tokenization with Mistral tokenization
- Tokenize a dataset into binary files
- Build a GPT-style transformer from scratch
- Train a small language model in Colab
- Use mixed precision training
- Save and reload checkpoints
- Generate short stories from prompts
- Run everything without Google Drive

---

## Current Final Setup

```text
Dataset: TinyStories
Tokenizer: mistralai/Mistral-7B-v0.1
Vocabulary size: 32000
Context length: 2048
Model type: GPT-style decoder-only transformer
Storage: Colab local /content only
Google Drive: Not used
Cost: Free
```

---

## Limitations

This is an experimental small language model.

It should not be expected to produce high-quality stories immediately.

The model may generate weak or incoherent stories if:

- It is trained for too few steps
- The model size is too small
- The validation loss is still high
- Sampling settings are too random
- The generation function does not stop at EOS

This project is mainly useful for understanding tokenizer impact, SLM training, and GPT-style model implementation.

---

## Future Work

Possible next steps:

- Add EOS-based stopping during generation
- Add loss curves using Matplotlib
- Compare Mistral tokenizer with GPT-2 tokenizer
- Train with `block_size = 512`, `1024`, and `2048`
- Try different model sizes
- Add inference script
- Convert notebook into Python files
- Push trained checkpoints to Hugging Face
- Later adapt the pipeline for a domain-specific nutrition SLM

---

## License

This repository is for educational and experimental purposes.

The TinyStories dataset and Mistral tokenizer should be used according to their respective licenses and terms.
