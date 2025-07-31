#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: wang
@time: 2025/7/31 9:31
@desc: 
"""
from torch import nn

from transformer.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        #
        self.encoder = nn.Sequential(
            Encoder(),
        )
        self.decoder = Decoder()

    def forward(self, x): # x [batch_size, seq_len, hidden_size]

        # 编码器
        x = self.encoder(x)

        # 解码器
        x = self.decoder(x)
        return x
