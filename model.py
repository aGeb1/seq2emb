import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """It's SHA right now, it'll be MLA later."""

    def __init__(self, dim, head_dim):
        super().__init__()

        self.wq = nn.Linear(dim, head_dim, bias=False)
        self.wk = nn.Linear(dim, head_dim, bias=False)
        self.wv = nn.Linear(dim, head_dim, bias=False)
        self.wo = nn.Linear(head_dim, dim, bias=False)

        self.scale = head_dim**0.5

    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        weight = q @ k.transpose(-2, -1) / self.scale
        weight = F.softmax(weight, dim=-1)
        output = weight @ v

        return self.wo(output)


class FeedForwardNetwork(nn.Module):
    """SwiGLU."""

    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()

        self.W = nn.Linear(dim, hidden_dim, bias=False)
        self.V = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.W2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        h = F.silu(self.W(x)) * self.V(x)
        h = self.dropout(h)
        out = self.W2(h)

        return out


class TransformerLayer(nn.Module):
    """One attention FFN pair in an encoder-only model."""

    def __init__(self, dim, head_dim, hidden_dim, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm()
        self.attention = Attention(dim, head_dim)
        self.dropout1 = nn.Dropout()

        self.norm2 = nn.LayerNorm()
        self.ffn = FeedForwardNetwork(dim, hidden_dim, dropout)
        self.dropout2 = nn.Dropout()

    def forward(self, x):
        h = x + self.dropout1(self.attention(self.norm1(x)))
        out = h + self.dropout2(self.ffn(self.norm2(h)))

        return out


class Transformer(nn.Module):
    """Encoder-only sequence to sequene component."""

    def __init__(self, num_layers, dim, head_dim, hidden_dim, dropout):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerLayer(dim, head_dim, hidden_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x  # In most models it would layer norm would be applied, but here it goes directly into another transformer.


class Seq2Emb(nn.Module):
    def __init__(self, vocab_size, dim, num_layers, head_dim, hidden_dim, dropout):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, dim)
        self.encoder = Transformer(num_layers, dim, head_dim, hidden_dim, dropout)
        self.decoder = Transformer(num_layers, dim, head_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.unembed = nn.Linear(dim, vocab_size, bias=False)


    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens)
        x = self.encoder(x)
        x[:, 1:] = torch.zero_like(x[:, 1:])  # Remove all embeddings besides CLS
        x = self.decoder(x)
        x = self.unembed(x)
        return x
