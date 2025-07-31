#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/7/31 9:34
@desc: 
"""
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        self.attention = Attention()
        self.ffn = FeedForward()

    def forward(self, x):
        x = self.attention(x)
        x = self.ffn(x)
        return x