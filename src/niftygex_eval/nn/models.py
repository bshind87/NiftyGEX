"""Neural models for the option-chain forecast: DeepSets, attention, attention + GRU."""
import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    """Per-contract MLP: F features -> d_model."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        return self.net(tokens)


class MaskedMeanPool(nn.Module):
    """Masked mean over contracts plus a log-count channel, projected back to d_model."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model + 1, d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # x: [*, N, d]   mask: [*, N]  (True = real contract)
        count = mask.sum(dim=-1).float()
        kept = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))
        mean = kept.sum(dim=-2) / count.clamp(min=1).unsqueeze(-1)
        with_count = torch.cat([mean, torch.log1p(count).unsqueeze(-1)], dim=-1)
        return self.proj(with_count)

class DeepSetsModel(nn.Module):
    """Step 4: embed the last snapshot, masked mean pool, then the head."""

    def __init__(self, n_token_features: int, n_state_features: int, config: dict) -> None:
        super().__init__()
        d = config["d_model"]
        self.embed = TokenEmbedding(n_token_features, d)
        self.pool = MaskedMeanPool(d)
        self.head = nn.Sequential(
            nn.Linear(d + n_state_features, config["head_hidden"]),
            nn.GELU(),
            nn.Linear(config["head_hidden"], 1),
        )

    def forward(self, tokens: Tensor, mask: Tensor, state: Tensor, return_attention: bool = False):
        last_tokens, last_mask = tokens[:, -1], mask[:, -1]
        chain = self.pool(self.embed(last_tokens), last_mask)
        out = self.head(torch.cat([chain, state], dim=-1)).squeeze(-1)
        return (out, None) if return_attention else out


def build_model(name: str, n_token_features: int, n_state_features: int, config: dict) -> nn.Module:
    """Return the named model; unknown names raise ValueError."""
    if name == "deepsets":
        return DeepSetsModel(n_token_features, n_state_features, config)
    raise ValueError(f"unknown model: {name}")