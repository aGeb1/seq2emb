import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """It's SHA right now, it'll be MLA later."""

    def __init__(self, dim, head_dim, base=10_000):
        super().__init__()

        self.wq = nn.Linear(dim, head_dim, bias=False)
        self.wk = nn.Linear(dim, head_dim, bias=False)
        self.wv = nn.Linear(dim, head_dim, bias=False)
        self.wo = nn.Linear(head_dim, dim, bias=False)

        self.scale = head_dim**0.5

        freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(200)  # Arbitrarily large
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer('freqs_cis', freqs_cis)


    def apply_rope(self, x):
        """The class embedding isn't rotated because it thinks it's too cool to be positional."""
        x_ = torch.view_as_complex(x[:, 1:].reshape(*x[:, 1:].shape[:-1], -1, 2))
        x[:, 1:] = torch.view_as_real(x_ * self.freqs_cis[:x_.shape[1]]).flatten(2)

        return x


    def forward(self, x):
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q, k = self.apply_rope(q), self.apply_rope(k)

        weight = q @ k.transpose(-2, -1) / self.scale
        weight = F.softmax(weight, dim=-1)
        output = weight @ v

        return self.wo(output)


class FeedForwardNetwork(nn.Module):
    """SwiGLU."""

    def __init__(self, dim, hidden_dim):
        super().__init__()

        self.W = nn.Linear(dim, hidden_dim, bias=False)
        self.V = nn.Linear(dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        h = F.silu(self.W(x)) * self.V(x)
        out = self.W2(h)

        return out


class TransformerLayer(nn.Module):
    """One attention FFN pair in an encoder-only model."""

    def __init__(self, dim, head_dim, hidden_dim):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attention = Attention(dim, head_dim)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim, hidden_dim)

    def forward(self, x):
        h = x + self.attention(self.norm1(x))
        out = h + self.ffn(self.norm2(h))

        return out


class Transformer(nn.Module):
    """Encoder-only sequence to sequene component."""

    def __init__(self, num_layers, dim, head_dim, hidden_dim):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerLayer(dim, head_dim, hidden_dim)
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
        self.encoder = Transformer(num_layers, dim, head_dim, hidden_dim)
        self.decoder = Transformer(num_layers, dim, head_dim, hidden_dim)
        self.norm = nn.LayerNorm(dim)
        self.unembed = nn.Linear(dim, vocab_size)

    
    def random_like(_, x):
        return torch.normal(0, 1, x.shape)

    def forward(self, tokens: torch.Tensor):
        x = self.embed(tokens)
        x = self.encoder(x)
        x[:, 1:] = self.random_like(x[:, 1:])  # Remove all embeddings besides CLS
        x = self.decoder(x)
        x = self.unembed(x)
        return x
