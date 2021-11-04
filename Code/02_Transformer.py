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
        #Training samples
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)

        #Output, multiply queries with keys, with matrics of several dimensions
        #n->batch size
        #q-> query
        #h-> heads
        #d-> heads dim
        #k-> key length
        #l-> length, dimension we want to multiply across
        energy = torch.einsum("nqhd, nkhd-->nhqk", [queries, keys])
        #queries shape: (N, query_len, heads, heads_dim)
        #keys shape: (N, key_len, heads, heads_dim)
        #energy shape: (N, heads, query_len, hey_len)
        #torch.bmm

        #mask is a triangular matrix
        #"-1e20" -> minus infinity
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy/(self.embed_size**(1/2)), dim=3)

        out = torch.einsum("nhql, nlhd-->nqhd",[attention, values]).reshape(N, query_len, self.heads*self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim) then flatten last two dimensions

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
#Inputs
class Encoder(nn.Module):
    #hyperparameters of the model
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion = forward_expansion,
                )
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0,seq_length).expand(N, seq_length).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
