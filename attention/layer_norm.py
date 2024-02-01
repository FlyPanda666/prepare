import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12) -> None:
        super().__init__()
        self.gamma = nn.Parameter(data=torch.ones(d_model))
        self.beta = nn.Parameter(data=torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


if __name__ == "__main__":
    x = torch.rand(size=(2, 3))
    ln = LayerNorm(d_model=3)
    y = ln(x)
    print(x)
