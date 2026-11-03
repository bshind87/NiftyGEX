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


class AttentionPool(nn.Module):
    """Learned-query attention pooling over contracts plus a log-count channel."""

    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.heads, self.d_head = heads, d_model // heads
        self.query = nn.Parameter(torch.randn(heads, self.d_head) * 0.02)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model + 1, d_model)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        # x: [*, N, d]   mask: [*, N]  (True = real contract)
        *lead, n, d = x.shape
        k = self.key(x).view(*lead, n, self.heads, self.d_head)
        v = self.value(x).view(*lead, n, self.heads, self.d_head)
        scores = torch.einsum("...nhe,he->...hn", k, self.query) / self.d_head**0.5
        pad = ~mask.unsqueeze(-2)
        weights = torch.softmax(scores.masked_fill(pad, -1e9), dim=-1).masked_fill(pad, 0.0)
        pooled = torch.einsum("...hn,...nhe->...he", weights, v).reshape(*lead, d)
        count = mask.sum(dim=-1).float()
        out = self.proj(torch.cat([pooled, torch.log1p(count).unsqueeze(-1)], dim=-1))
        return out, weights


class ChainEncoder(nn.Module):
    """Embed contracts, apply self-attention with the padding mask, then attention-pool each snapshot."""

    def __init__(self, n_token_features: int, config: dict) -> None:
        super().__init__()
        d = config["d_model"]
        self.embed = TokenEmbedding(n_token_features, d)
        layer = nn.TransformerEncoderLayer(
            d, config["heads"], config["ff_dim"], config["dropout"], batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, config["layers"], enable_nested_tensor=False)
        self.pool = AttentionPool(d, config["heads"])

    def forward(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        # tokens: [M, N, F]   mask: [M, N]  (True = real contract)
        ignore = ~mask
        empty = ~mask.any(dim=-1)
        ignore[:, 0] = ignore[:, 0] & ~empty
        x = self.encoder(self.embed(tokens), src_key_padding_mask=ignore)
        return self.pool(x, mask)


def make_head(n_in: int, hidden: int) -> nn.Module:
    """Prediction head: n_in -> hidden -> 1."""
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, 1))


class DeepSetsModel(nn.Module):
    """Step 4: embed the last snapshot, masked mean pool, then the head."""

    def __init__(self, n_token_features: int, n_state_features: int, config: dict) -> None:
        super().__init__()
        d = config["d_model"]
        self.embed = TokenEmbedding(n_token_features, d)
        self.pool = MaskedMeanPool(d)
        self.head = make_head(d + n_state_features, config["head_hidden"])

    def forward(self, tokens: Tensor, mask: Tensor, state: Tensor, return_attention: bool = False):
        last_tokens, last_mask = tokens[:, -1], mask[:, -1]
        chain = self.pool(self.embed(last_tokens), last_mask)
        out = self.head(torch.cat([chain, state], dim=-1)).squeeze(-1)
        return (out, None) if return_attention else out


class AttentionModel(nn.Module):
    """Step 5: self-attention over the last snapshot, attention pooling, then the head."""

    def __init__(self, n_token_features: int, n_state_features: int, config: dict) -> None:
        super().__init__()
        self.chain = ChainEncoder(n_token_features, config)
        self.head = make_head(config["d_model"] + n_state_features, config["head_hidden"])

    def forward(self, tokens: Tensor, mask: Tensor, state: Tensor, return_attention: bool = False):
        chain, weights = self.chain(tokens[:, -1], mask[:, -1])
        out = self.head(torch.cat([chain, state], dim=-1)).squeeze(-1)
        return (out, weights.unsqueeze(1)) if return_attention else out


class AttentionGRUModel(nn.Module):
    """Step 6: self-attention on every snapshot, then a GRU over the chain vectors in time order."""

    def __init__(self, n_token_features: int, n_state_features: int, config: dict) -> None:
        super().__init__()
        self.chain = ChainEncoder(n_token_features, config)
        self.gru = nn.GRU(config["d_model"], config["gru_hidden"], batch_first=True)
        self.head = make_head(config["gru_hidden"] + n_state_features, config["head_hidden"])

    def forward(self, tokens: Tensor, mask: Tensor, state: Tensor, return_attention: bool = False):
        b, l, n, f = tokens.shape
        chain, weights = self.chain(tokens.reshape(b * l, n, f), mask.reshape(b * l, n))
        _, hidden = self.gru(chain.view(b, l, -1))
        out = self.head(torch.cat([hidden[-1], state], dim=-1)).squeeze(-1)
        return (out, weights.view(b, l, -1, n)) if return_attention else out


def build_model(name: str, n_token_features: int, n_state_features: int, config: dict) -> nn.Module:
    """Return the named model; unknown names raise ValueError."""
    models = {"deepsets": DeepSetsModel, "attention": AttentionModel, "attention_gru": AttentionGRUModel}
    if name not in models:
        raise ValueError(f"unknown model: {name}")
    return models[name](n_token_features, n_state_features, config)