import torch
import torch.nn as nn
from encoder_layer import EncoderLayer
from transformer_embedding import TransformerEmbedding


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        max_len,
        dropout,
        base,
        n_head,
        ffn_hidden,
        n_layers,
        device,
    ) -> None:
        super().__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
            device=device,
            base=base,
        )
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    n_head=n_head,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, src_mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


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

    encoder = Encoder(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len,
        dropout=dropout,
        base=base,
        n_head=n_head,
        ffn_hidden=ffn_hidden,
        n_layers=n_layers,
        device=device,
    )

    batch_size = 3
    seq_len = 20
    x = torch.randint(low=1, high=80, size=(batch_size, seq_len))
    print(x)
    y = encoder(x)
    print(y.shape)
    assert y.shape == (batch_size, seq_len, d_model)
