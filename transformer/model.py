#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/7/31 10:17
@desc: 
"""
import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x+self.pe[:, :x.size(1)].to(x.device)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        x = (x - mean) / (torch.sqrt(var + self.eps))
        return self.a_2 * x + self.b_2

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "embed_dim must be divisible by num_heads"

        self.scale = math.sqrt(self.head_dim)

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        bsz, seq_len, dim = x.size()
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x) # [bsz, seq, hsz]

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        k = k.transpose(2, 3)
        q_k = torch.matmul(q, k)/self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(q_k)
            q_k = q_k.masked_fill(mask, -1e9)

        scores = self.softmax(q_k) # [bsz, n_head, seq, seq]
        atten = torch.matmul(scores, v).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.w_o(atten)

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, n_head, ffn_size, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(hidden_size, n_head)
        self.layer_norm1 = LayerNorm(hidden_size)
        self.layer_norm2 = LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size),
        )

    def forward(self,x, mask=None):
        h = self.self_attn(x, mask=mask)
        h = self.layer_norm1(x + h)
        ffn_h = self.feed_forward(h)
        out = self.layer_norm2(h + ffn_h)
        return out

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        return x

class Encoder(nn.Module):
    def __init__(self, hidden_size, n_head, max_len, voc_size, ffn_size, dropout, n_layers, device):
        super().__init__()

        self.token_embeddings = TokenEmbedding(vocab_size=voc_size, embed_size=hidden_size)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(hidden_size, n_head, ffn_size, dropout) for _ in range(n_layers)]
        )

    def forward(self,x, mask=None):
        h = self.token_embeddings(x)
        for layer in self.encoder_layers:
            h = layer(h, mask)
        return h

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, max_len, enc_voc_size, hidden_size, n_head, ffn_size, n_layers, drop_prop, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx

        self.encoder = Encoder(hidden_size=hidden_size,
                               n_head=n_head,
                               voc_size=enc_voc_size,
                               max_len=max_len,
                               ffn_size=ffn_size,
                               dropout=drop_prop,
                               n_layers=n_layers,
                               device=device
                               )

    def forward(self, src, trg) -> torch.Tensor:
        src_pad_mask = self.src_pad_mask(src)
        # trg_pad_mask = self.src_pad_mask(trg)

        enc_h = self.encoder(src, src_pad_mask)
        # trg_out = self.decoder(trg, enc_h, trg_pad_mask)
        # return trg_out

    def src_padding_mask(self, src):
        return (src==self.src_pad_idx)

    def trg_causal_mask(self, trg):
        seq_len = trg.size(-1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask