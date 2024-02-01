import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device="cpu", base=10000) -> None:
        super().__init__()
        self.embedding = torch.zeros(
            size=(max_len, d_model), device=device, requires_grad=False
        )
        _2i = torch.arange(start=0, end=d_model, step=2, device=device).float()
        pos = torch.arange(start=0, end=max_len, device=device).float()
        pos = pos.unsqueeze(dim=1)
        # pos的shape就是[max_len, 1]
        # _2i的shape就是[d_model//2]
        self.embedding[:, 0::2] = torch.sin(pos / (base ** (_2i / d_model)))
        self.embedding[:, 1::2] = torch.cos(pos / (base ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.shape
        return self.embedding[:seq_len, :]


def drawing():
    plt.figure(figsize=(15, 5))
    pe = PositionalEmbedding(d_model=24, max_len=100, device="cpu", base=100)
    y = pe(torch.zeros(1, 100))
    plt.plot(np.arange(100), y[:, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()


if __name__ == "__main__":
    x = torch.rand(size=(2, 4))
    pe = PositionalEmbedding(d_model=24, max_len=10, device="mps", base=10000)
    y = pe(x)
    print(y.shape)
    assert y.shape == (4, 24)
    drawing()
