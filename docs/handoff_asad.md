# Task 4: the neural models (for Asad)

**Goal:** write the three neural models from the proposal (slides 12 and 14, steps 4–6) as plain PyTorch modules.

**Done when:** `tests/test_nn_models.py` passes.

**What you need:** only the repo and the environment. No project data is needed: the tests use random tensors shaped like the real token history.

**What happens after:** once the tests pass, Bhalchandra connects your models to the real data and the walk-forward harness, and runs the training. That part (dataset, trainer, runs) is not part of this task.

## 1. Setup (about 15 minutes)

```bash
git clone <repo URL>            # or GitHub Desktop: File → Clone repository
cd NiftyGEX
conda env create -f env/environment.yml    # Python 3.11 with PyTorch 2.10 (conda-forge build)
conda activate niftygex
PYTHONPATH=src pytest -q                   # everything passes; tests/test_nn_models.py is skipped until your file exists
```

Work on a branch, for example `asad/models`, and open a pull request when the tests pass. Bhalchandra reviews and merges it.

## 2. Where your code goes

| File | Owner | Contents |
|---|---|---|
| `src/niftygex_eval/nn/models.py` | **you** | The modules and `build_model` |
| `src/niftygex_eval/nn/dataset.py` | Bhalchandra | Real tokens → tensors (not needed for this task) |
| `src/niftygex_eval/nn/trainer.py` | Bhalchandra | Training loop, early stopping, seeds |
| `tests/test_nn_models.py` | given | Your acceptance tests; please don't edit them, and ask if one seems wrong |

## 3. The interface (exact)

```python
def build_model(name: str, n_token_features: int, n_state_features: int, config: dict) -> torch.nn.Module:
    """name is "deepsets", "attention" or "attention_gru"; any other name raises ValueError."""
```

Every model's `forward(tokens, mask, state, return_attention=False)` takes:

| Argument | Shape and type | Meaning |
|---|---|---|
| `tokens` | `[B, L, N, F]` float32 | B days, L snapshots (5 minutes apart), N contract slots, F ≈ 15 features per contract. **Index −1 along L is the 15:15 prediction snapshot**; index 0 is the oldest |
| `mask` | `[B, L, N]` bool | True means a real contract, False means padding. Padded slots can hold any values, including garbage |
| `state` | `[B, S]` float32 | Volatility-state features of the day (IV, realized volatility, calendar, total GEX); S ≈ 10–15 |

It returns `Tensor[B]`: one number per day, the predicted log volatility ratio. With `return_attention=True` it returns `(output, weights)`, where `weights` are the **pooling** attention weights with shape `[B, L_used, heads, N]`:
- `L_used` is 1 for `attention` and L for `attention_gru`;
- the weights sum to 1 over real contracts and are 0 on padding;
- `deepsets` may return `None` for the weights.

The `config` dict has these defaults (they will live in `configs/eval.yaml`):

```python
{"d_model": 64, "heads": 4, "layers": 2, "ff_dim": 128, "dropout": 0.1, "gru_hidden": 64, "head_hidden": 32}
```

## 4. The three models

All three share the same building blocks; they differ only in which ones they switch on. That's what RQ2 tests.

| Model (proposal step) | Reads | Contracts interact? | History? |
|---|---|---|---|
| `deepsets` (4) | Last snapshot only | No: per-contract MLP, then pooling | No |
| `attention` (5) | Last snapshot only | Yes: self-attention | No |
| `attention_gru` (6) | All L snapshots | Yes | Yes: GRU over the L chain vectors |

The building blocks, in order:
1. **Token embedding.** A small MLP per contract: F → 64 → 64.
2. **Self-attention** (`attention` and `attention_gru` only). 2 encoder layers, 4 heads, d = 64, feed-forward width 128, dropout 0.1, using the padding mask.
   - **No positional encoding.** A contract's position comes from its `log_moneyness` feature, and the output must not depend on the order of contracts.
   - `torch.nn.TransformerEncoderLayer(..., batch_first=True, norm_first=True)` is fine.
3. **Pooling to one 64-d chain vector per snapshot.**
   - For `deepsets`, a masked mean over contracts.
   - For the attention models, attention pooling: one learned query per head, a masked softmax over contracts, and the heads concatenated back to 64-d.
   - **In both cases, also append log(1 + number of real contracts) and project back to 64.** Softmax pooling and means are averages, so without this count channel a total like GEX can't be represented. A test checks it.
4. **GRU** (`attention_gru` only). Hidden size 64 over the L chain vectors in time order; use the last hidden state.
5. **Head.** Concatenate the summary (the chain vector, or the GRU state) with `state`, then MLP → 32 → 1.

The target for `attention_gru` is about 100k parameters. The test accepts 70k–150k.

## 5. Rules
- **No data loading and no file I/O in `models.py`:** tensors in, tensors out.
- **Use the device of the input tensors**, so the model works on CPU and on Apple-silicon `mps`. Use no global state; seeds are set by the trainer.
- **Edge cases must stay finite** (the tests check each):
  - a snapshot can have **zero** real contracts (data gaps); the output must stay finite, with no NaN from a softmax over an empty set;
  - padded values must not change the output;
  - shuffling contracts must not change the output.
- **Code style:** small, typed functions and classes with one-line docstrings, matching `src/niftygex_eval/forecast/`.

## 6. Acceptance tests

```bash
PYTHONPATH=src pytest tests/test_nn_models.py -q
```

| Test | What it checks |
|---|---|
| output shape | One finite number per sample, for all three models |
| shuffling | Shuffling contracts within a snapshot leaves the output unchanged (eval mode) |
| padding | Garbage in padded slots, or extra padded slots, leaves the output unchanged |
| empty snapshot | A sample with no real contracts still gives a finite output |
| count channel | n copies of the same contracts and 2n copies give different outputs |
| history | `deepsets` and `attention` ignore earlier snapshots; `attention_gru` uses them |
| size | `attention_gru` has 70k–150k parameters |
| attention weights | Shape `[B, L_used, heads, N]`, summing to 1 over real contracts, 0 on padding |
| overfitting | Each model can fit 16 random targets (a sanity check that gradients flow) |
| names | An unknown model name raises `ValueError` |

The full file runs in a few seconds on a laptop CPU.

## 7. Timeline and check-in
- **Check-in on 5 November:** share which tests pass, even if not all do.
- **Target: pull request merged by 8 November.** Training runs start on 9 November.
- If something in the spec is unclear or a test looks wrong, message Bhalchandra early rather than working around it.

## 8. After this task

Next for you:
- **Attention maps and case studies (task 8).** Once the models are trained, you'll get the derived data folders through Drive (`data/features/`, `data/targets/`, about 520 MB); you won't need the raw data.
- **Writing.** Drafts of the related-work and method sections, and the final slides.

## Background reading in the repo

- `README.md`: the project and the run order.
- `docs/findings_data.md`: what the token history looks like, including a one-day heatmap of exactly what your model reads.
- `docs/findings_forecast.md`: the classical baselines your model has to beat (HAR-RV, mean IC 0.434).
- `notebooks/02_pipeline_and_features.ipynb`: one day followed through every stage.
