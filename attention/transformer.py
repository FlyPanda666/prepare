import torch
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder


class Transformer(nn.Module):
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
        src_pad_idx,
        tgt_pad_idx,
        tgt_sos_idx,
    ) -> None:
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.device = device
        self.encoder = Encoder(
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
        self.decoder = Decoder(
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

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, src_mask=src_mask, tgt_mask=tgt_mask)
        return output

    def make_src_mask(self, src):
        """_summary_

        Args:
            src (_type_): batch_size * seq_len
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # print("src_mask:", src_mask.shape)
        print("src_mask", src_mask)
        return src_mask

    def make_tgt_mask(self, tgt):
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(3)
        print("trg_pad_mask: ", tgt_pad_mask.shape)
        tgt_len = tgt.shape[1]
        tgt_sub_mask = (
            torch.tril(torch.ones(tgt_len, tgt_len))
            .type(torch.ByteTensor)
            .to(self.device)
        )
        # print("tgt_sub_mask", tgt_sub_mask.shape)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        tgt_mask = tgt_mask.to(torch.bool)
        print("tgt_mask", tgt_mask)
        # print("tgt_mask:", trg_mask.shape)
        return tgt_mask


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
    src_pad_idx = 1
    tgt_pad_idx = 1
    tgt_sos_idx = 0
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        max_len=max_len,
        dropout=dropout,
        base=base,
        n_head=n_head,
        ffn_hidden=ffn_hidden,
        n_layers=n_layers,
        device=device,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        tgt_sos_idx=tgt_sos_idx,
    )

    batch_size = 3
    seq_len = 20
    x = torch.randint(low=1, high=80, size=(batch_size, seq_len))
    print(x)
    y = model(x, x)
    print(y.shape)
    assert y.shape == (batch_size, seq_len, vocab_size)
