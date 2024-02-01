import torch
import torch.nn as nn
import math


class ScaleDotProductAttention(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        batch_size, head, seq_len, d_model = k.size()
        k_t = k.transpose(2, 3)
        score = torch.matmul(q, k_t) / math.sqrt(d_model)
        if mask is not None:
            score = score.masked_fill(mask=mask, value=-10000)
        score = self.softmax(score)
        v = torch.matmul(score, v)
        return v, score


if __name__ == "__main__":
    q = torch.rand(size=(2, 8, 12, 512))
    k = q
    v = q
    attention = ScaleDotProductAttention()
    out = attention(q, k, v)
    print(out[0].shape)
    print(out[1].shape)
    assert out[0].shape == (2, 8, 12, 512)
    assert out[1].shape == (2, 8, 12, 12)
