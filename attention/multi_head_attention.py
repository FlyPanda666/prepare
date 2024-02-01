import torch
import torch.nn as nn

from scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head) -> None:
        super().__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_concat = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        v, score = self.attention(q, k, v, mask)
        v = self.concat(v)
        v = self.w_concat(v)
        return v, score

    def split(self, tensor: torch.Tensor):
        batch_size, seq_len, d_model = tensor.size()
        d_head = d_model // self.n_head
        tensor = tensor.view(batch_size, seq_len, self.n_head, d_head).transpose(1, 2)
        return tensor

    def concat(self, tensor: torch.Tensor):
        batch_size, n_head, seq_len, d_head = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return tensor


if __name__ == "__main__":
    q = torch.rand(size=(2, 12, 512))
    k = q
    v = q
    attention = MultiHeadAttention(d_model=512, n_head=8)
    out = attention(q, k, v)
    print(out[0].shape)
    print(out[1].shape)
    assert out[0].shape == (2, 12, 512)
    assert out[1].shape == (2, 8, 12, 12)
