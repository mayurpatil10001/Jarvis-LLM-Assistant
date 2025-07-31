import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
from collections import Counter
import math
import os
from tqdm import tqdm

class TokenizerVocab:
    def __init__(self):
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<START>", 3: "<END>"}
        self.vocab_size = 4
    
    def build_vocab(self, texts, min_freq=2):
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size += 1
    
    def encode(self, text):
        words = text.lower().split()
        return [self.word_to_idx.get(word, 1) for word in words]
    
    def decode(self, indices):
        return " ".join([self.idx_to_word.get(idx, "<UNK>") for idx in indices])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.scale = math.sqrt(self.d_k)
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, mask=None):
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class JARVISModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=8, d_ff=2048, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        x = self.ln_f(x)
        return self.head(x)
    def generate(self, tokenizer, prompt, max_length=50, temperature=0.8, top_k=50):
        self.eval()
        tokens = [2] + tokenizer.encode(prompt)  # Add START token
        with torch.no_grad():
            for _ in range(max_length):
                input_tensor = torch.tensor([tokens[-512:]]).long().to(next(self.parameters()).device)
                logits = self.forward(input_tensor)
                logits = logits[0, -1, :] / temperature
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                if next_token == 3:
                    break
                tokens.append(next_token)
        response_tokens = [t for t in tokens[1:] if t != 3]
        return tokenizer.decode(response_tokens)

