"""Acceptance tests for the neural models (src/niftygex_eval/nn/models.py); see docs/handoff_asad.md.

They use random tensors shaped like the real token history, so they need no project data. They are skipped
until niftygex_eval.nn.models exists.
"""
import pytest

torch = pytest.importorskip("torch")
models = pytest.importorskip("niftygex_eval.nn.models")

NAMES = ["deepsets", "attention", "attention_gru"]
CONFIG = {"d_model": 64, "heads": 4, "layers": 2, "ff_dim": 128, "dropout": 0.1, "gru_hidden": 64, "head_hidden": 32}
B, L, N, F, S = 4, 12, 30, 15, 10


def batch(b: int = B, l: int = L, n: int = N, seed: int = 0):
    """Random tokens with a ragged padding mask (True = real contract); every snapshot keeps at least 5 tokens."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randn(b, l, n, F, generator=g)
    counts = torch.randint(5, n + 1, (b, l), generator=g)
    mask = torch.arange(n).view(1, 1, n) < counts.unsqueeze(-1)
    state = torch.randn(b, S, generator=g)
    return tokens, mask, state


def build(name: str, **overrides):
    torch.manual_seed(0)
    model = models.build_model(name, n_token_features=F, n_state_features=S, config={**CONFIG, **overrides})
    return model.eval()


@pytest.mark.parametrize("name", NAMES)
def test_output_is_one_finite_number_per_sample(name: str) -> None:
    out = build(name)(*batch())
    assert out.shape == (B,) and torch.isfinite(out).all()


@pytest.mark.parametrize("name", NAMES)
def test_shuffling_contracts_leaves_the_output_unchanged(name: str) -> None:
    model, (tokens, mask, state) = build(name), batch()
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.allclose(model(tokens, mask, state), model(tokens[:, :, perm], mask[:, :, perm], state), atol=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_padding_values_are_ignored(name: str) -> None:
    model, (tokens, mask, state) = build(name), batch()
    garbage = tokens.clone()
    garbage[~mask] = 1e3 * torch.randn(int((~mask).sum()), F)
    wider = torch.cat([garbage, 50 * torch.randn(B, L, 7, F)], dim=2)
    wider_mask = torch.cat([mask, torch.zeros(B, L, 7, dtype=torch.bool)], dim=2)
    with torch.no_grad():
        assert torch.allclose(model(tokens, mask, state), model(wider, wider_mask, state), atol=1e-5)


@pytest.mark.parametrize("name", NAMES)
def test_an_empty_snapshot_gives_a_finite_output(name: str) -> None:
    tokens, mask, state = batch()
    mask[0] = False  # sample 0 has no tokens at any snapshot (a data gap)
    out = build(name)(tokens, mask, state)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", NAMES)
def test_the_number_of_contracts_matters(name: str) -> None:
    """Pooling keeps a count channel: n copies of the same contracts and 2n copies must not look identical."""
    tokens, _, state = batch(b=1, n=8)
    mask = torch.ones(1, L, 8, dtype=torch.bool)
    doubled, doubled_mask = torch.cat([tokens, tokens], dim=2), torch.ones(1, L, 16, dtype=torch.bool)
    model = build(name)
    with torch.no_grad():
        assert not torch.allclose(model(tokens, mask, state), model(doubled, doubled_mask, state), atol=1e-6)


@pytest.mark.parametrize("name,uses_history", [("deepsets", False), ("attention", False), ("attention_gru", True)])
def test_only_attention_gru_reads_earlier_snapshots(name: str, uses_history: bool) -> None:
    model, (tokens, mask, state) = build(name), batch()
    earlier = tokens.clone()
    earlier[:, :-1] += 3.0 * torch.randn_like(earlier[:, :-1])  # index -1 is the 15:15 prediction snapshot
    with torch.no_grad():
        same = torch.allclose(model(tokens, mask, state), model(earlier, mask, state), atol=1e-6)
    assert same is not uses_history


def test_attention_gru_has_about_100k_parameters() -> None:
    count = sum(p.numel() for p in build("attention_gru").parameters())
    assert 70_000 <= count <= 150_000, count


@pytest.mark.parametrize("name,snapshots", [("attention", 1), ("attention_gru", L)])
def test_pooling_attention_weights_are_returned_for_attention_maps(name: str, snapshots: int) -> None:
    tokens, mask, state = batch()
    out, weights = build(name)(tokens, mask, state, return_attention=True)
    heads = CONFIG["heads"]
    assert out.shape == (B,) and weights.shape == (B, snapshots, heads, N)
    used = mask[:, -snapshots:].unsqueeze(2).expand_as(weights)
    assert torch.allclose(weights.sum(-1), torch.ones(B, snapshots, heads), atol=1e-5)
    assert (weights[~used] == 0).all()


@pytest.mark.parametrize("name", NAMES)
def test_a_tiny_batch_can_be_overfit(name: str) -> None:
    tokens, mask, state = batch(b=16, l=4, n=12, seed=3)
    target = torch.randn(16, generator=torch.Generator().manual_seed(4))
    model = build(name, dropout=0.0).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(400):
        optimizer.zero_grad()
        loss = torch.mean((model(tokens, mask, state) - target) ** 2)
        loss.backward()
        optimizer.step()
    assert float(loss) < 0.1 * float(target.var())


def test_unknown_model_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        models.build_model("transformer_xl", n_token_features=F, n_state_features=S, config=CONFIG)
