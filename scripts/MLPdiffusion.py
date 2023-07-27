#!/usr/bin/env python3

import torch
import torch.nn as nn
from typing import Tuple, Optional

class MLPDiffusion(nn.Module):

    def __init__(self, dim:int,dout:int,num_steps: int,hidd_units:Optional[int] = 128):
        super().__init__()

        self.linears = nn.ModuleList([
            nn.Linear(dim, hidd_units),
            nn.ReLU(),
            nn.Linear(hidd_units, hidd_units),
            nn.ReLU(),
            nn.Linear(hidd_units, hidd_units),
            nn.ReLU(),
            nn.Linear(hidd_units, dout)
        ])

        self.step_embeddings = nn.ModuleList([
            nn.Embedding(num_steps, hidd_units),
            nn.Embedding(num_steps, hidd_units),
            nn.Embedding(num_steps, hidd_units),
        ])


    def forward(self,x:torch.Tensor,t:torch.Tensor):
        for idx, embedding_layer in enumerate(self.step_embeddings):
            # print("The size of t before embedding is:", t.size())
            t_embedding = embedding_layer(t)
            # print("t_embedding size is:",t_embedding.shape)
            x = self.linears[2*idx](x)
            # print("x size is:", x.shape)
            x += t_embedding
            x = self.linears[2*idx+1](x)

        x = self.linears[-1](x)

        return x
