import torch
import torch.nn as nn
from decoder_layer import DecoderLayer
from transformer_embedding import TransformerEmbedding


class Decoder(nn.Module):
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
                DecoderLayer(
                    d_model=d_model,
                    n_head=n_head,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, tgt, enc, src_mask=None, tgt_mask=None):
        tgt = self.embedding(tgt)
        for layer in self.layers:
            tgt = layer(tgt, enc, src_mask, tgt_mask)
        output = self.linear(tgt)
        return output


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

    encoder = Decoder(
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
    y = encoder(x, None)
    print(y.shape)
    assert y.shape == (batch_size, seq_len, vocab_size)
