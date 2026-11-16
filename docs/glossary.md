# Glossary

Terms used across the code, reports and status page. Definitions follow the implementation; `docs/schema.md` and
`docs/evaluation.md` give the exact column-level rules.

## 1. Market and option basics

| Term | Meaning | In this project |
|---|---|---|
| **NIFTY** | India's main stock index (50 large companies) | The underlying studied |
| **Spot** | The index's current price | Scales GEX; reference for walls and levels |
| **Futures / front-month future** | A contract to buy the index at a set price on a future date; the front month is the nearest one | Its 5-minute prices measure realized volatility; one contract per day, never switched intraday |
| **Option (call / put)** | A call pays if the index ends above the **strike** K; a put pays if it ends below | Each contract at a snapshot is one token for the neural models |
| **Expiry** | The date an option ends | Tokens use each day's nearest expiry with data, excluding a same-day expiry |
| **Option chain** | All strikes, calls and puts of an expiry at one moment | About 70 contracts at 15:15 |
| **Forward (F)** | The market-implied index price at expiry, from put-call parity | Reference point for moneyness and IV |
| **Moneyness / log-moneyness** | How far a strike is from the forward: log(K / F); 0 = at the money | Token field; x-axis of the attention maps |
| **ATM / OTM** | At the money (strike ≈ forward) / out of the money (a call above or a put below the forward) | IV for both legs of a strike comes from the OTM leg |
| **Open interest (OI)** | Number of option contracts still open | Reported in index units, so GEX needs no lot multiplier |
| **OI change %** | Change in OI since the previous close, in percent | Token field `oi_change_since_previous_close_pct`; not log-transformed |
| **Volume** | Contracts traded so far today | Token field `volume_today` |
| **Lot size** | Index units per contract | Keyed by contract expiry in `configs/regimes.yaml`; checked against OI divisibility |
| **Participant OI** | NSE's daily open positions by trader type (clients, pros, FIIs, DIIs) | Source of the inferred dealer sign; lagged one trading day |
| **Weekly expiry rationalisation** | The November 2024 NSE rule change that cut weekly index expiries | The before/after regime marker for RQ3 |

## 2. Volatility

| Term | Meaning | In this project |
|---|---|---|
| **Volatility** | How much a price moves, regardless of direction | What is forecast |
| **RV, realized volatility** | Volatility that actually happened: √(sum of squared within-day 5-minute log returns); the overnight gap is excluded | Next-day front-month futures RV is the target's numerator |
| **Annualized** | Scaled to a per-year figure so days and years are comparable | Feature columns `*_annualized*`; cancels in the target ratio |
| **Trailing 20-day mean RV** | Mean daily RV over the 20 trading days ending on the snapshot day (at least 15 present) | The target's denominator: the "normal" level |
| **Target, y** | log(next-day RV / trailing 20-day mean RV); 0 = a normal day, +0.22 ≈ 25% more volatile | The outcome every model predicts (`log_rv_ratio_trailing`) |
| **Intraday RV so far** | Today's futures RV up to the snapshot | Feature `rv_intraday_so_far_annualized` |
| **HAR terms** | Mean RV over the last 1, 5 and 20 trading days, ending the previous day | Features `rv_trailing_{1,5,20}d_annualized_prev` |
| **IV, implied volatility** | Volatility expected by the option market, backed out of option prices with Black-76 | Token field; ATM IV in the state vector |
| **India VIX** | NSE's 30-day implied-volatility index | Previous day's close is a feature; gaps stay NaN |
| **Skew (25-delta)** | IV of 25-delta puts minus 25-delta calls: the price of crash protection | State feature `skew_25d` |
| **Term slope** | IV of a later expiry minus a nearer one | State feature `term_slope` |

## 3. Greeks and GEX

| Term | Meaning | In this project |
|---|---|---|
| **Black-76** | The standard option-pricing formula on a forward | Gives IV, delta and gamma |
| **Delta** | Option price change per 1-point index move | Token field |
| **Gamma** | Rate at which delta changes as the index moves | Core of GEX; token field |
| **Time to expiry** | Calendar-year time from `available_at` to 15:30 IST on expiry day | Token field; Greeks need it > 0 |
| **GEX, gamma exposure** | gamma × OI × spot² × 0.01: rupee hedging flow per 1% index move | The one-number chain summary the project tests |
| **Dealers / hedging** | Market makers hedge their option books. Long gamma (positive GEX) means buying dips and selling rallies, which damps moves; short gamma means chasing moves, which amplifies them | The theory behind GEX |
| **GEX naive** | Sign convention: calls +, puts − (dealers assumed long calls, short puts) | `gex_total_naive`, `strike_gex_naive_crore` |
| **GEX absolute** | Every leg counted as +: pure size | `gex_total_absolute`, `strike_gex_absolute_crore` |
| **GEX inferred** | Call and put legs signed by the Pro participants' net index-call and index-put positions (previous day) | `gex_total_inferred`, `gex_sign_inferred` |
| **Normalized GEX** | Total GEX / causal trailing mean of absolute total GEX | `gex_norm_*`; removes the years-long growth in OI |
| **Strike GEX** | GEX of a single contract, in crore rupees | Token fields; the neural models see GEX strike by strike |
| **Call wall / put wall** | Strike with the largest call OI above spot / put OI below spot | Level features (distance in % of spot) |
| **Max pain** | Settlement strike that minimizes total intrinsic payout to option holders | Level feature |
| **Gamma flip** | Index level where total GEX changes sign | Level feature `flip_naive_distance_pct`; `has_flip` |
| **GEX concentration (HHI)** | Herfindahl index of absolute GEX across strikes | Feature `gex_concentration_hhi` |

## 4. Models: the forecasting ladder

| Term | Meaning |
|---|---|
| **Ladder (steps 0–6)** | The models in order of complexity, all scored on the same rows and walks |
| **Historical mean** (step 0) | Forecasts the training-window mean every day; IC undefined, an RMSE/QLIKE reference |
| **Persistence** (step 0) | Yesterday's RV relative to its 20-day level |
| **HAR-RV** (step 1) | Heterogeneous Autoregressive model of RV: OLS on the 1-, 5- and 20-day RV terms plus today's RV so far. A strong, hard-to-beat baseline |
| **Ridge regression** | Linear regression with a penalty shrinking coefficients; HAR + GEX (step 2), alpha tuned on the inner split |
| **LightGBM** (step 3) | Gradient-boosted decision trees on about 40 summary features, tuned on the inner split |
| **Token** | One option contract at one snapshot, described by 15 fields (`neural.token_features`) |
| **Padding mask** | Marks which contract slots are real; chains differ in size |
| **State vector** | 14 day-level inputs (IV, skew, term slope, RVs, previous VIX, days to expiry, expiry-day flag, normalized GEX, inferred sign) given to every neural model |
| **DeepSets** (step 4) | Encodes each contract separately, then takes a masked mean: reads every contract, no interactions |
| **Self-attention** (step 5) | Each contract attends to every other (Transformer encoder); one snapshot, attention pooling |
| **Heads / layers** | Parallel attention views (4) and stacked encoder blocks (2) |
| **Attention pooling** | A learned weighted average turning the contracts of a snapshot into one vector |
| **GRU** | Gated Recurrent Unit: a recurrent network that reads a sequence in order |
| **Attention + GRU** (step 6) | Self-attention per snapshot, then a GRU over the last L snapshots (L = 12, the last hour): the project's model |
| **Window (L)** | Number of 5-minute snapshots of history the GRU reads; 24 and 72 in the window ablation |
| **Parameters** | Learned weights: attention + GRU 112,257; self-attention 87,297; DeepSets 11,969 |

## 5. Training

| Term | Meaning |
|---|---|
| **Huber loss** | Squared error for small misses, linear for large ones (δ = 1); the neural training loss |
| **AdamW** | Adam optimizer with decoupled weight decay (lr 1e-3, decay 0.01) |
| **Epoch / batch** | One pass over the training rows / the 64 days per update |
| **Early stopping / patience** | Stop when the inner-validation loss has not improved for 8 epochs; keep the best epoch |
| **Overfitting** | Fitting training noise rather than signal |
| **Seeds** | Random initializations (11, 12, 13); forecasts are averaged across them |
| **Refit** | Retraining on the full training window after tuning; off for the neural models (`refit: false`) |
| **Scaler** | Per-feature mean and standard deviation from the walk's training days only; missing values and padding become 0 |
| **Signed log** | sign(x) · log(1 + \|x\|) for heavy-tailed token fields (`neural.signed_log_features`) |
| **MPS / CPU** | Apple GPU backend / processor; attention models train on MPS, DeepSets on CPU |

## 6. Evaluation design

| Term | Meaning |
|---|---|
| **Snapshot** | The moment a forecast is made: 15:15 IST each trading day |
| **Bar / start-labeled** | A 5-minute bar labeled t covers [t, t + 5 min) and is known at t + 5 min |
| **`available_at`** | When a row would really have been known; a feature at snapshot T uses only rows with `available_at` ≤ T |
| **`target_available_at`** | When the target is known: 15:30 on the next trading day |
| **Leakage** | Using information not available at the snapshot; prevented by `available_at` rules and tests |
| **Lag** | Position in the token history: lag 0 = 15:15, lag 11 = 14:20 |
| **Walk-forward** | Train on the past, test on the next block, move forward; no shuffling. 26 walks |
| **Expiry cycle** | The span between consecutive weekly expiries; walks train on at least 52 cycles and test on blocks of 13 |
| **Embargo** | A one-cycle gap between training and test |
| **Purge** | Dropping training rows whose `target_available_at` falls after the test block starts |
| **Inner split** | Validation cycles carved from the end of the training window for tuning and early stopping; test blocks are never used |
| **`check_split`** | Harness check that train and test never overlap or leak; must pass for every walk |
| **Pre-registration** | Tests and win rules written in `configs/eval.yaml` before any neural run |
| **Exploratory** | Anything not pre-registered: reported, but not as confirmation |
| **Ablation** | Changing one part (window length, a token field) to measure what it contributes |

## 7. Scores and statistics

| Term | Meaning |
|---|---|
| **IC, Information Coefficient** | Spearman rank correlation of forecasts and outcomes within a test block; 1 = perfect order, 0 = random. Primary score, averaged over walks |
| **Spearman correlation** | Correlation of ranks rather than values; robust to outliers |
| **RMSE** | Root mean squared error of the forecast; lower is better |
| **QLIKE** | The standard loss for volatility forecasts. It compares realized variance s with forecast variance h: s/h − log(s/h) − 1 (on our log ratio, exp(2(y − ŷ)) − 2(y − ŷ) − 1); 0 is perfect, lower is better. Why we use it: it punishes under-forecasting more (a forecast 20% too low costs about twice as much as one 20% too high; `docs/findings_models.md` shows the computed values), it measures misses as ratios so calm and wild days count alike, and it ranks forecasts correctly even though true volatility is only measured with noise (Patton, 2011) |
| **R² vs HAR** | Share of HAR's squared error a model removes; negative means worse than HAR |
| **95% confidence interval** | Range that plausibly contains the true value; an interval for a difference that spans 0 shows no clear difference |
| **p-value** | How surprising the result would be if there were no true difference; the threshold is α = 0.05 |
| **Paired t-test** | Compares two models on the same walks using per-walk differences |
| **Walks won** | Number of the 26 blocks where model A beats model B |
| **Diebold–Mariano (DM)** | Test of whether two forecasts' daily losses differ; negative loss difference favours model A |
| **Newey–West** | Variance correction for autocorrelated daily loss differences, used inside DM |
| **Holm correction** | Step-down adjustment of p-values within a family of tests (RQ2) |
| **Intersection–union rule** | RQ1 win rule: attention + GRU must beat every classical rung, each at p < 0.05 |
| **Welch t-test** | Two-group comparison allowing unequal sizes and variances (RQ3: 19 walks before, 7 after the marker) |
| **Relative weight** | A contract's pooling weight × the number of contracts in its snapshot; 1 = an equal share |
| **Effective share** | (1 / Σw²) / number of contracts; 1 = all weighted equally, small = concentrated on a few |

## 8. Research questions and bookkeeping

| Term | Meaning |
|---|---|
| **RQ1** | Does reading every contract (attention + GRU) forecast better than every classical model? |
| **RQ2** | If so, why: cross-contract interactions (self-attention vs DeepSets) or history (attention + GRU vs self-attention)? |
| **RQ3** | Does the gain over tuned LightGBM survive the November 2024 rule change? |
| **Manifest / config hash** | Each stage records a fingerprint of its settings and inputs; downstream stages refuse to run on a mismatch |
| **Run id / `git_commit` / `git_dirty`** | Each results-store run records its id, the code commit and whether the working tree had uncommitted edits |
| **Results store** | `outputs/results/`: one directory per run (`run.json`, predictions, walk metrics, attention) plus `index.jsonl` |
| **Parquet** | Columnar file format for all data stages |
