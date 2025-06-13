# Transformer Architecture Overview
<p align="center">
<img src="https://www.comet.com/site/wp-content/uploads/2023/07/Screen-Shot-2023-07-11-at-9.48.50-PM-1024x769.png" width = 550 >
</p>

The Transformer architecture consists of an encoder and a decoder, but this implementation focuses only on the decoder part, which is commonly used in models like GPT.

# Decoder-Only Transformer Model (GPT-like) Implementation

## Overview
This repository contains a from-scratch implementation of a Decoder-Only Transformer (GPT-like) model in PyTorch. The code is modular and easy to follow, making it suitable for both research and educational purposes. The model is ideal for tasks such as language modeling, text generation, and sequence prediction.

## Model Components

### 1. InputEmbedding
- Converts input tokens into dense vectors of a specified model dimension.
- Utilizes the `nn.Embedding` layer for efficient lookup and scaling.

### 2. PositionalEncoding
- Injects positional information into the token embeddings so the model can capture word order.
- Uses sine and cosine functions to generate unique encodings for each position.

### 3. MultiHeadAttention
- Implements the multi-head self-attention mechanism, allowing the model to focus on different parts of the sequence simultaneously.
- Employs masked attention to prevent the model from attending to future tokens during training (causal masking).
- Operates over query, key, and value matrices for each attention head.

### 4. FeedForwardNetwork
- Applies a two-layer fully connected network to each position independently.
- Uses ReLU activation for non-linearity and dropout for regularization.

### 5. Residual Connections
- Adds skip connections and layer normalization to stabilize training and improve gradient flow.
- Includes dropout to prevent overfitting.

### 6. GPT Class
- Integrates all the above components into a model.
- Supports stacking multiple Transformer blocks (n_block) for increased depth and expressiveness.
- Outputs predictions through a final linear layer mapping to the vocabulary size.

### 7. Config
- Uses a `dataclass` to manage and organize model hyperparameters (e.g., vocab_size, batch_size, model_dimension, n_heads, etc.) for easy configuration and reproducibility.

## Use Cases
- Language modeling
- Text completion and generation
- Sequence data processing
- Educational purposes for understanding Transformer internals

## References
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/pdf/1706.03762)
- [GPT: Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
