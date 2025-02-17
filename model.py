import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size:int, model_dimension:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = model_dimension
        self.embedding = nn.Embedding(self.vocab_size,self.d_model)
    def forward(self, x):
        # x = (batch_size, seq_len)
        x = self.embedding(x) * math.sqrt(self.d_model)
        return x # (batch_size, seq_len, d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int,model_dimension: int,  dropout: float):
        super().__init__()
        self.seq_len = max_seq_len
        self.d_model = model_dimension
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(max_seq_len, self.d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1],:]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self,model_dimension:int,n_heads:int, dropout:float, max_seq_len:int):
        super().__init__()
        self.d_model = model_dimension
        self.n_heads = n_heads
        self.head_dim = model_dimension // n_heads
        self.dropout = nn.Dropout(dropout)
        
        assert model_dimension % n_heads == 0 ,"Model dimension should be divisible by n_heads"

        self.q_W = nn.Linear(self.d_model, self.d_model, bias=False) # Weights of the query matrix
        self.k_W = nn.Linear(self.d_model, self.d_model, bias=False) # Weights of the key matrix
        self.v_W = nn.Linear(self.d_model, self.d_model, bias=False) # Weights of the value matrix
        self.o_W = nn.Linear(self.d_model, self.d_model, bias=False) # Weights of the output matrix
        
        self.register_buffer("mask",torch.tril(torch.ones(max_seq_len,max_seq_len)).view(1,1,max_seq_len,max_seq_len))

    def forward(self,x):
        q = self.q_W(x) # (batch, seq_len, d_model) x (batch, d_model, d_model) --> (batch, seq_len, d_model)
        k = self.k_W(x) # (batch, seq_len, d_model) x (batch, d_model, d_model) --> (batch, seq_len, d_model)
        v = self.v_W(x) # (batch, seq_len, d_model) x (batch, d_model, d_model) --> (batch, seq_len, d_model)

        q = q.view(q.shape[0], q.shape[1], self.n_heads, self.head_dim).transpose(1,2) # (batch_size, seq_len, self.n_heads, self.head_dim) --> (batch_size, self.n_heads, seq_len, self.head_dim)
        k = k.view(k.shape[0], k.shape[1], self.n_heads, self.head_dim).transpose(1,2) # (batch_size, seq_len, self.n_heads, self.head_dim) --> (batch_size, self.n_heads, seq_len, self.head_dim)
        v = v.view(k.shape[0], v.shape[1], self.n_heads, self.head_dim).transpose(1,2) # (batch_size, seq_len, self.n_heads, self.head_dim) --> (batch_size, self.n_heads, seq_len, self.head_dim)

        attention_score = (q @ torch.transpose(k,-2,-1)) / math.sqrt(self.head_dim) # (batch_size, self.n_heads, seq_len, self.head_dim) x # (batch_size, self.n_heads, self.head_dim, seq_len) --> (batch_size, self.n_heads, seq_len, seq_len)
        attention_score.masked_fill_(self.mask[:,:,:x.shape[1],:x.shape[1]] == 0, -torch.inf) # (batch_size, self.n_heads, seq_len, seq_len)
        attention_score = self.dropout(F.softmax(attention_score,dim=-1)) # (batch_size, self.n_heads, seq_len, seq_len)
        attention_out = attention_score @ v # (batch_size, self.n_heads, seq_len, seq_len) x (batch_size, self.n_heads, seq_len, self.head_dim) --> (batch_size, self.n_heads, seq_len, self.head_dim)
        attention_out = torch.transpose(attention_out,2,1).contiguous().view(attention_score.shape[0], -1, self.n_heads*self.head_dim) # (batch_size, seq_len, d_model)
        
        return self.o_W(attention_out) # (batch_size, seq_len, d_model)
    


class FeedForwardNetwork(nn.Module):
    def __init__(self,model_dimension:int, hidden_layer:int, dropout:float):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(model_dimension,hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer,model_dimension),
            nn.Dropout(dropout)),
    def forward(self,x):
        return self.ffn(x)
    

class Resudial_Connections(nn.Module):
    def __init__(self, dropout:float,model_dimension:int):
        super().__init__()
        self.layernormalize = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        return x + self.dropout(self.layernormalize(x))
    
# Hyperparameters
@dataclass
class Config:
    vocab_size:int  
    batch_size:int 
    max_seq_len:int 
    model_dimension:int
    n_heads:int 
    dropout:float
    n_block : int

class GPT(nn.Module):
    def __init__(self,vocab_size:int, model_dimension:int, max_seq_len:int, dropout:float,  n_heads:int,hidden_layer:int, n_block:int):
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size = vocab_size, model_dimension = model_dimension)
        self.positinal_encoding = PositionalEncoding(max_seq_len = max_seq_len, model_dimension = model_dimension, dropout = dropout)
        self.multi_head_attention = MultiHeadAttention(model_dimension = model_dimension, n_heads = n_heads,dropout=dropout, max_seq_len=max_seq_len)
        self.forward_propagation = FeedForwardNetwork(model_dimension = model_dimension, hidden_layer=hidden_layer, dropout=dropout)
        self.res_connect_0 = Resudial_Connections(dropout=dropout,model_dimension=model_dimension)
        self.res_connect_1 = Resudial_Connections(dropout=dropout,model_dimension=model_dimension)
        self.out = nn.Linear(model_dimension,vocab_size)
        self.n_block = n_block
    def forward(self,x):
        x = self.input_embedding(x) 
        x = self.positinal_encoding(x)
        for _ in range(self.n_block):
            x = x + self.multi_head_attention(self.res_connect_0(x))
            x = x + self.forward_propagation(self.res_connect_1(x))
        x =  self.out(x)
        return x