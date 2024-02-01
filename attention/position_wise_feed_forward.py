import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, ffn_hidden, d_model, dropout) -> None:
        super().__init__()
        self.linner1 = nn.Linear(in_features=d_model, out_features=ffn_hidden)
        self.linner2 = nn.Linear(in_features=ffn_hidden, out_features=d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linner1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linner2(x)
        return x


if __name__ == "__main__":
    x = torch.rand(size=(2, 12, 512))
    pw = PositionWiseFeedForward(512 * 4, 512, 0.8)
    out = pw(x)
    print(out.shape)
    assert out.shape == (2, 12, 512)
