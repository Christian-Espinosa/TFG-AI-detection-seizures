import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        #parts = heads, if there are 256 embeded size in 8 parts, 32 parts
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads_dim = embed_size // heads #integer division

        assert(self.heads_dim*heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.keys = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = nn.Linear(self.heads_dim, self.heads_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.heads_dim, embed_size)


    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)