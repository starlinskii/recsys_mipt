import torch
import torch.nn as nn


class FactorizationMachine(nn.Module):
    def __init__(self, num_features, factors=16):
        super().__init__()
        self.linear = nn.Embedding(num_features, 1)
        self.factors = nn.Embedding(num_features, factors)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.factors.weight, std=0.01)

    def forward(self, feature_ids):
        linear_score = self.linear(feature_ids).sum(dim=1).squeeze(-1)
        factor_values = self.factors(feature_ids)
        factor_sum = factor_values.sum(dim=1)
        pair_score = 0.5 * (
            factor_sum.square() - factor_values.square().sum(dim=1)
        ).sum(dim=1)
        return self.bias + linear_score + pair_score
