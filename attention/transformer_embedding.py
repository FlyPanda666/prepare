import torch
import torch.nn as nn
from token_embedding import TokenEmbedding
from positional_embedding import PositionalEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(
        self, vocab_size, d_model, max_len, dropout, device, base: int = 10000
    ) -> None:
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.positional_embedding = PositionalEmbedding(
            d_model=d_model, max_len=max_len, device=device, base=base
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_embedding(x)
        return self.dropout(token_embedding + positional_embedding)


if __name__ == "__main__":
    vocab_size = 100
    d_model = 512
    max_len = 1000
    dropout = 0.5
    base = 10000
    n_head = 8
    ffn_hidden = 512 * 4
    n_layers = 12
    device = "cpu"

    te = TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len,
        dropout=dropout,
        base=base,
        device=device,
    )

    batch_size = 3
    seq_len = 20
    x = torch.randint(low=1, high=80, size=(batch_size, seq_len))
    print(x)
    y = te(x)
    print(y.shape)
    assert y.shape == (batch_size, seq_len, d_model)
