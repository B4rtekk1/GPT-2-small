# GPT-2 small

## Introduction

This project implements a simplified version of the GPT-2 architecture, adapted for character-level modeling on the Tiny Shakespeare dataset. Below is a detailed description of the model's architecture and its components.

## Architecture

### Configuration (`GPT2Config`)

- **block_size**: Maximum sequence length (context window) for the model.
- **vocab_size**: Number of unique tokens (characters) in the dataset.
- **n_layer**: Number of Transformer blocks (layers).
- **n_head**: Number of self-attention heads per block.
- **d_model**: Dimensionality of the model (embedding and hidden size).
- **dropout**: Dropout probability for regularization.
- **d_ff**: Size of the feed-forward layer (default: 4 × d_model).
- **activation_function**: Nonlinearity used in the feed-forward network (e.g., GELU).

You can easily configure your model by editing this file [config.py](config.py)

### Token & Position Embeddings

- **Token Embedding**: Maps each character to a dense vector of size `d_model`.
- **Position Embedding**: Adds positional information to each token, allowing the model to distinguish between different positions in the sequence.

### Transformer Block (repeated `n_layer` times)

Each block consists of:

- **Multi-Head Self-Attention**

  - Splits the input into `n_head` heads.
  - Each head computes scaled dot-product attention independently.
  - Outputs are concatenated and projected back to `d_model`.
  - Causal masking is applied to prevent attending to future tokens.
- **Layer Normalization**: Applied before and after attention and feed-forward sublayers.
- **Feed-Forward Network**
  - Two linear layers with an activation function in between (e.g., GELU).
  - First layer expands to `d_ff`, then projects back to `d_model`.
- **Residual Connections**: Add input to the output of each sublayer.
- **Dropout**: Applied after attention and feed-forward outputs for regularization.

### Output Head

- **LayerNorm**: Fibal normalization of hidden states.
- **Linear Head**: Projects the final hidden state to the vocabulary size for next-token preditcion.
- **Weight Typing**: The ouput head shares weights with the input token embedding for efficiency.

## Training Details

- **Loss**: Cross-entropy loss is computed between the predicted logits and target sequence (shifted by one character).
- **Optimizer**: AdamW is used for parameter updates.
- **Fixed Precision** Training supports FP16 via PyTorch AMP for faster compution and lower memory usage.
- **Multi-GPU**: The model supports multi-GPU training using `torch.nn.DataParallel`.

## Generation

- **Autoregressive Sampling**: The model generates text one token at a time, feeding its own predictions back as input.
- **Temperature & Top-k Sampling**: Controls randomness and diversity of generated text.
- **Early Stopping**: Generation stops if the end-of-text token is produced.

## 5. File Structure

- `model.py`: Model and block definitions.
- `config.py`: Model configuration class.
- `train.py`: Training loop and dataset handling.
- `gpt2.ipynb`: Jupyter notebook version for interactive use.

## 6. Example Hyperparameters

- `block_size`: 128
- `n_layer`: 4
- `n_head`: 4
- `d_model`: 128
- `dropout`: 0.1
