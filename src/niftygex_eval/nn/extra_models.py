"""Exploratory neural models (configs/eval_exploratory_v2.yaml): state-only MLP, hybrid and multi-day attention + GRU.

They share the input contract of nn/models.py: ``forward(tokens [B, L, N, F], mask [B, L, N], state [B, S])``.
"""
import torch
import torch.nn as nn
from torch import Tensor

from niftygex_eval.nn.models import AttentionGRUModel, ChainEncoder, build_model, make_head


class StateMLPModel(nn.Module):
    """Volatility state only: state -> d_model -> head. Tokens are ignored (the E1 state-only control)."""

    def __init__(self, n_token_features: int, n_state_features: int, config: dict) -> None:
        super().__init__()
        d = config["d_model"]
        self.body = nn.Sequential(nn.Linear(n_state_features, d), nn.GELU(), nn.Dropout(config["dropout"]))
        self.head = make_head(d, config["head_hidden"])

    def forward(self, tokens: Tensor, mask: Tensor, state: Tensor, return_attention: bool = False):
        out = self.head(self.body(state)).squeeze(-1)
        return (out, None) if return_attention else out


class MultiDayAttentionGRUModel(nn.Module):
    """Attention + GRU over today's last L snapshots plus a second GRU over the previous K days' 15:15 chains.

    The first ``multiday_days`` snapshots of the input are the previous days (oldest first), the rest the intraday
    window. Both go through one shared chain encoder; a day without a chain enters its GRU as a zero vector.
    """

    def __init__(self, n_token_features: int, n_state_features: int, config: dict) -> None:
        super().__init__()
        self.days = int(config["multiday_days"])
        self.chain = ChainEncoder(n_token_features, config)
        self.gru = nn.GRU(config["d_model"], config["gru_hidden"], batch_first=True)
        self.daily_gru = nn.GRU(config["d_model"], config["gru_hidden"], batch_first=True)
        self.head = make_head(2 * config["gru_hidden"] + n_state_features, config["head_hidden"])

    def forward(self, tokens: Tensor, mask: Tensor, state: Tensor, return_attention: bool = False):
        b, l, n, f = tokens.shape
        chain, weights = self.chain(tokens.reshape(b * l, n, f), mask.reshape(b * l, n))
        chain = chain.view(b, l, -1) * mask.any(dim=-1, keepdim=True).float()
        _, daily = self.daily_gru(chain[:, : self.days])
        _, intraday = self.gru(chain[:, self.days:])
        out = self.head(torch.cat([intraday[-1], daily[-1], state], dim=-1)).squeeze(-1)
        return (out, weights.view(b, l, -1, n)) if return_attention else out


def build_extra_model(name: str, n_token_features: int, n_state_features: int, config: dict) -> nn.Module:
    """Exploratory models by name; the main models (deepsets, attention, attention_gru) come from nn/models.py.

    ``hybrid_attention_gru`` is the attention + GRU architecture; the hybrid part (HAR forecast as an input and
    output = HAR forecast + correction) is in forecast/residual.py.
    """
    extra = {"state_mlp": StateMLPModel, "hybrid_attention_gru": AttentionGRUModel, "multiday_attention_gru": MultiDayAttentionGRUModel}
    if name in extra:
        return extra[name](n_token_features, n_state_features, config)
    return build_model(name, n_token_features, n_state_features, config)
