# Bid Optimization Engine

## Setup & Running

**Requires Python 3.9+**

```bash
git clone <repo-url>
pip install -r requirements.txt
python main.py
```

Outputs `bid_recommendations.csv` — one recommended bid per keyword.  
No API key needed — Claude metadata is pre-populated in `keyword_metadata_cache.json`.  
If running with different data, set `ANTHROPIC_API_KEY` in your environment for full LLM classification of new keywords; without it, unknown keywords fall back to mid-range defaults and the algorithm still runs.  
Full walkthrough with code, visualisations, and decisions: `jupyter notebook bid_optimizer.ipynb`

---

## Project Structure

```
bid-optimizer/
├── main.py                # Entry point
├── bid_optimizer.py       # Core algorithm module
├── bid_optimizer.ipynb    # Full DS walkthrough: EDA, decisions, implementation
├── campaign_data.csv      # 30 days of keyword performance data
├── requirements.txt
└── README.md
```

---

## 1. Task Objective & Business Context

### What Are We Building

Feedvisor manages advertising campaigns for brands on Amazon. The optimizer sets a **maximum cost-per-click (CPC) bid** per keyword — a standing instruction that stays in effect until the next daily cycle.

**Input:** `campaign_data.csv` — 30 days of daily keyword-level data (~200 keywords, 10 campaigns).  
**Output:** One recommended bid per unique keyword for the next daily cycle.

The bid is a lever with two opposing forces:
- Too low → lost auctions, fewer impressions, no revenue
- Too high → expensive clicks, high spend per sale, low ROAS

### Business Constraints — Priority Order

**1. Budget — hard ceiling, non-negotiable.**  
Total projected daily spend per campaign must never exceed the campaign's daily budget. EDA shows all 10 campaigns are currently 1.6×–4.8× over budget. Enforcing this is the primary job.

**2. ROAS — the north-star metric.**  
ROAS = Revenue / Spend. Maximize it. No target to hit — higher is always better. Budget is the only ceiling.

**3. Bid stability — advertiser trust.**  
No keyword changes by more than ±50% per cycle. Erratic bid swings erode advertiser confidence faster than any optimisation mistake.

> **Constraint hierarchy:** Budget (1) overrides stability (3). When the budget slash loop must drive a keyword's bid to the marketplace floor ($0.20–$0.26), it will go below `current_avg_bid × 0.50` — this is intentional and declared, not a violation. Every such keyword is flagged in the `reason` column.

**4. Explainability.**  
Every recommendation includes a human-readable `reason` column.

**5. Graceful handling of sparse data.**  
Keywords with thin history must receive principled estimates — not silence.

---

## 2. How Business Constraints Shape the Algorithm

### 2.1 Portfolio Problem, Not Pointwise Prediction

**All keywords in a campaign share the same daily budget pool.** Setting keyword A's bid independently of keywords B, C, D is wrong by construction — the sum will exceed the budget and any subsequent uniform cut destroys the efficiency signal.

**The correct framing:** given a fixed daily budget per campaign, distribute it across keywords in proportion to each keyword's efficiency, so that the most profitable keywords receive more spend and the least profitable receive less.

> Bids are not predicted in isolation — they are derived from how much budget each keyword *deserves* relative to its campaign peers.

### 2.2 Full 30-Day Window — No Train/Test Split

A train/test split is correct for forecasting problems where we estimate generalisation to future time. This is not a forecasting problem. We are generating one operational recommendation per keyword for the *next* daily cycle. The right inputs are the best available performance estimates — which means using all 30 days, not discarding the most recent 9 (the 30% most valuable signal) to create an artificial holdout.

Model quality is measured by whether the budget constraint is satisfied and whether projected ROAS improves — not held-out prediction accuracy.

### 2.3 Asymmetric Budget Allocation — Not Uniform Scaling

The naive budget fix when a campaign overspends by 3× is to divide all bids by 3. This penalises high-ROAS cash-cows identically to money-losing keywords — a fundamental failure.

**The correct approach:** each keyword receives a share of the campaign's daily budget proportional to its efficiency score. High-score keywords get more; low-score keywords get less. The bid follows mechanically:

```
score_weight_k   = keyword_score_k / sum(keyword_scores in campaign)
allocated_spend  = score_weight_k × campaign_daily_budget
recommended_bid  = allocated_spend / avg_daily_clicks_k
```

Budget constraint is satisfied by construction, before any individual bid is computed.

### 2.4 LLM Integration — Load-Bearing for Sparse Keywords

For data-rich keywords, a purely behavioural efficiency score (ROAS, CVR, CTR) works well. For keywords with thin or zero history, behavioural features are undefined — a purely data-driven system silently assigns them zero budget forever.

**Hybrid efficiency score:**

```
keyword_score = (confidence × behavioral_score)
              + ((1 − confidence) × llm_specificity_score)
              + llm_margin_modifier
              + llm_funnel_modifier
```

Where `llm_specificity_score` is the embedding-derived centroid distance (how semantically specific the keyword is), and the modifiers are small additive terms from Claude's structured metadata.

- `confidence = 1` → 100% behavioural signal, LLM has zero weight
- `confidence = 0` → 100% LLM prior, sole allocation signal for zero-history keywords

The LLM is not decoration. Remove it and every sparse keyword gets zero budget regardless of semantic potential. "Noise cancelling headphones" (high commercial intent) would be treated identically to "what is bluetooth" (informational).

### 2.5 Architecture Summary

| Decision | Reason |
|---|---|
| Full 30-day window, no split | Operational optimisation — use best available signal |
| Portfolio allocation, not pointwise | Keywords share a budget pool — isolation violates the constraint |
| Asymmetric score-weighted allocation | Protects cash-cows, cuts money-losers — symmetric scaling destroys value |
| Hybrid behavioural + LLM score | LLM is primary signal for sparse/cold-start keywords |
| ±50% stability cap | Business trust — erratic changes erode advertiser confidence |

> Full business reasoning and derivation: **Section 1–2** in [bid_optimizer.ipynb](bid_optimizer.ipynb)

---

## 3. Data Investigation — Questions & Findings

Before implementing anything, the data must answer specific questions that calibrate the algorithm. Each question flows from a design decision above.

### Q1 — Missing Data: Random or Informative?

**Finding:** No structural missing values. 186 of 200 keywords appear all 30 days; 2 keywords appear ~3 days. 54% of rows have zero revenue — valid negative signal (keyword ran, nobody converted), not missing data.

**Decision:** Aggregate over `days_active` (not calendar days). A keyword that ran 10 of 21 days at $5/day has `avg_daily_spend = $5`, not $50/21 = $2.38. The underestimate propagates directly into budget enforcement errors.

### Q2 — Conversion Sparsity: How Much Own-Data ROAS Signal Do We Have?

**Finding:** 87% of keywords have ≥10 conversions over 30 days. Only 3 keywords (2%) have zero conversions. 99% have ≥100 clicks. This is a data-rich portfolio.

**Decision:** Own-data ROAS is reliable for the vast majority. The LLM prior is a fallback for the 3% edge cases — not the main mechanism. Confidence thresholds of 30 clicks and 7 active days yield confidence ≈ 1 for nearly the entire portfolio, which is correct. Also rules out DNN: 200 keyword-level observations is far too small for neural networks on tabular data.

### Q3 — Bid Variability: Does `current_bid` Change?

**Finding:** 96% of keywords have 3+ unique bid values over 30 days — bids are actively adjusted. Only 7 keywords have static bids.

**Decision:** Despite variation, 3–5 distinct bid values is insufficient to learn a reliable per-keyword bid-response curve. Too sparse, too noisy (day-of-week effects, competitor changes). Confirms the portfolio allocation approach — derive bid from budget share, not from curve-fitting.

### Q4 — ROAS Distribution

**Finding:** Range 0.1–9.7, std > mean (high variance), median below 1 — most keywords are currently losing money or barely breaking even.

**Decision:** The high variance confirms the ±50% change cap — raw ROAS ratios would suggest 5×–10× bid changes that would destabilise campaigns. The wide spread means score-weighted allocation creates meaningful differentiation: top-ROAS keywords will receive substantially more budget.

### Q5 — Categorical Balance

**Finding:** Match types balanced (broad 28%, phrase 36%, exact 35%). Campaign sizes 11–27 keywords.

**Decision:** No special encoding or grouping needed.

### Q6 — Budget Utilisation (Most Critical Finding)

**Finding:** Every campaign is over its daily budget. Range: 1.6× (Beauty) to 4.8× (Pet Supplies, $572/day against a $120 budget). No exceptions.

**Decision:** Budget enforcement is the dominant output force. Every recommended bid will be substantially lower than the current bid. The accuracy of `avg_daily_clicks × recommended_bid` as projected spend is critical — this directly drives whether the budget constraint is met.

### Q7 — Temporal Trend in ROAS

**Finding:** Spearman r = 0.17, p = 0.36 — no statistically significant trend. Daily fluctuation is noise.

**Decision:** Flat 30-day aggregation is valid. No recency weighting needed.

> Full code, visualisations, and per-finding outputs: **Section 3** in [bid_optimizer.ipynb](bid_optimizer.ipynb)

---

## 4. From EDA to Algorithm — Confidence Score Design

The confidence score is the bridge between EDA findings and the hybrid score formula:

```
confidence = sqrt( min(total_clicks / 30, 1)  ×  min(days_active / 7, 1) )
```

- **Geometric mean** (not arithmetic): both click volume *and* temporal stability must hold. 500 clicks on a single day = near-zero temporal stability. The geometric mean captures this correctly.
- **Thresholds are fixed** (30 clicks, 7 days), not derived from dataset statistics — avoids any future leakage risk and anchors to business domain knowledge about minimum reliable sample sizes.
- **Q2 calibration:** 99% of keywords have ≥100 clicks and appear most/all 30 days → confidence ≈ 1 for nearly the entire portfolio → own-data ROAS dominates → correct.
- **Q1 calibration:** 2 keywords with ~3 active days → day factor ≈ 0.43 → confidence ≈ 0.65 → blended score that leans more on LLM prior → appropriate caution.

> Confidence score distributions by tier: **Section 4** in [bid_optimizer.ipynb](bid_optimizer.ipynb)

---

## 5. Model Architecture & Feature Engineering Plan

This section translates every EDA finding into a concrete engineering decision. Each stage, each feature, and each formula is grounded in a specific observation from Section 3.

### 5.0 Why a Score-Based Heuristic and Not a Classic ML Model

This is a deliberate choice driven by three hard constraints the EDA exposed — not a shortcut.

**Constraint 1 — No target variable (Q3)**
Supervised learning requires a label: the "correct" bid for each keyword. We never observe that. We only observe revenue at the bids that were *actually* used. The counterfactual — "what would revenue have been at bid $2.50 instead of $1.80?" — is unobserved. Without counterfactual outcomes, there is no target to train against.

**Constraint 2 — No bid-response curve (Q3)**
The natural proxy target is a per-keyword bid elasticity curve: how does ROAS change as bid changes? This requires bid variation. Q3 found 3–5 unique bid values per keyword over 30 days — far too sparse to fit a reliable curve. Fitting a regression on 3–5 points with day-to-day noise (competitor activity, seasonality) would produce confident but meaningless coefficients.

**Constraint 3 — Sample size rules out tabular ML (Q2)**
After aggregation, we have ~200 keyword-level observations. Gradient boosting and neural networks on tabular data require thousands of samples to generalise. At 200 rows, any model complex enough to capture non-linear ROAS interactions would overfit immediately. Ridge regression is technically feasible but would simply re-learn what the score formula already encodes — no new information.

**What the EDA pointed to instead**
The correct framing is a **portfolio allocation problem, not a prediction problem**. The question is not "predict the optimal bid" — it is "given a fixed budget, which keywords deserve more of it?" That question is answered by ranking, not by regression. A well-calibrated score — grounded in ROAS, CVR, and semantic signal — gives the ranking. The bid follows mechanically from the allocated budget share.

The LLM components (embedding specificity and Claude structured metadata) are where AI adds value that pure heuristics cannot: reading the raw keyword text to estimate commercial potential for keywords with no behavioural history. That is the load-bearing AI contribution — not wrapping a hand-coded formula in a neural network.

> Full stage-by-stage walkthrough with formula derivations: **Section 5** in [bid_optimizer.ipynb](bid_optimizer.ipynb)

```
[ Stage 1: Behavioral Feature Engineering    ]
                |
                v
[ Stage 2: Campaign Contextualization         ]
                |
                v
[ Stage 3: LLM Semantic Feature Layer         ]
                |
                v
[ Stage 4: Unified Keyword Score              ]
                |
                v
[ Stage 5: Asymmetric Portfolio Optimization  ]
                |
                v
[ Stage 6: Marketplace Compliance & Output    ]
```

---

### Stage 1 — Behavioral Feature Engineering

**Source:** EDA Q1, Q2, Q4, Q5, Q7

The raw dataset (~6,000 rows) is collapsed into one row per keyword — a flat profile matrix feeding every downstream stage.

**Critical rule from Q1:** all rates divide by `days_active = CountUnique(date)`, not by 30 calendar days. A keyword that ran 10 days and spent $50 total has `avg_daily_spend = $5.00`. Dividing by 30 gives $1.67 — a 3× underestimate that flows into Stage 5 and causes budget violations.

| Feature | Formula | EDA Justification |
|---|---|---|
| `historical_roas` | total_revenue / total_spend | Q2: 87% of keywords have ≥10 conversions — ROAS is a reliable signal |
| `cvr` | total_conversions / total_clicks | Q2: captures buyer quality independently of bid level |
| `ctr` | total_clicks / total_impressions | Ad relevance proxy — lower weight; partially bid-driven |
| `avg_daily_clicks` | total_clicks / days_active | Q1: informative absence rule; feeds projected spend in Stage 5 |
| `avg_daily_spend` | total_spend / days_active | Q1: same informative absence rule |
| `current_avg_bid` | mean(current_bid) over 30 days | Q7: no temporal trend — 30-day mean is stable; ±50% cap anchor |
| `match_type` | Ordinal: exact=1.0, phrase=0.8, broad=0.5 | Q5: balanced distribution; ordering reflects specificity |
| `confidence` | sqrt( min(clicks/30, 1) × min(days_active/7, 1) ) | Q1, Q2: geometric mean of click saturation × temporal stability |

**Why geometric mean for confidence?** A keyword with 500 clicks on a single day has saturation but no temporal stability. Arithmetic mean gives 0.5. Geometric mean gives near-zero — correct, because one day of data is not trustworthy. Both dimensions must hold simultaneously.

---

### Stage 2 — Campaign Contextualization

**Source:** EDA Q4 (ROAS std > mean)

Absolute ROAS cannot rank keywords across campaigns — a ROAS of 2.0 is a top performer in one campaign and a laggard in another. The fix: express each keyword's performance relative to its own budget pool.

```
campaign_avg_roas  =  sum(campaign revenue) / sum(campaign spend)
roas_vs_campaign   =  keyword historical_roas / campaign_avg_roas
```

- `roas_vs_campaign > 1` → outperforms campaign peers → deserves more budget
- `roas_vs_campaign < 1` → underperforms → candidate for reduction

This feature is what makes the scoring genuinely asymmetric across keywords within the same campaign.

---

### Stage 3 — LLM Semantic Feature Layer

**Source:** Business constraint 5 (sparse data); Q2 (3 zero-conversion keywords); architectural decision 2.4

Standard tabular features are structurally blind to commercial context. A tabular model cannot distinguish a high-margin luxury query from a low-margin commodity — they look identical until enough conversion data accumulates. The LLM layer reads the raw `keyword_text` to fill this gap.

**Component A — Embedding Layer (fully offline, no API key)**

Model: `all-MiniLM-L6-v2` via `sentence-transformers`. Runs locally, zero external dependencies.

Each `keyword_text` is encoded into a 384-dimensional dense semantic vector. These embeddings power two distinct features:

**A1 — Specificity Score (centroid distance)**

Compute the centroid (mean vector) of all keyword embeddings. Score each keyword by its cosine distance from that centroid:
- Far from centroid = semantically specific = long-tail, high-converting
- Close to centroid = generic head term = high competition, lower conversion efficiency

On Amazon, commercial intent is implicit in every search. The signal that differentiates keyword value is *specificity*, not intent. Centroid distance captures this with no hardcoded anchors.

**A2 — Neighbor ROAS (KNN in embedding space)**

For sparse keywords (low confidence), instead of relying solely on an LLM prior, we borrow the historical ROAS signal from semantically similar keywords that have rich data:

1. For each keyword, find its k=5 nearest neighbors by cosine similarity in embedding space
2. Compute similarity-weighted average ROAS from those neighbors:

```
neighbor_roas = sum(similarity_i × confidence_i × historical_roas_i) / sum(similarity_i × confidence_i)
```

**Why this matters:** a keyword with 3 days of history doesn't get a pure LLM estimate — it gets the actual earned ROAS of the 5 most semantically similar keywords in the portfolio, weighted by both similarity and each neighbor's confidence. A sparse neighbor with 2 clicks (low confidence, unreliable ROAS) contributes proportionally less than a data-rich neighbor. If all neighbors are sparse, the prior falls back to the portfolio mean. Specificity tells you *how long-tail* the keyword is; neighbor ROAS tells you *what similar, data-rich keywords actually earned*. Together they form a cold-start prior grounded in real data.

---

**Component B — Claude Structured Metadata (API, cached to JSON)**

Model: `claude-haiku-4-5-20251001`. Called once per unique keyword, cached to `keyword_metadata_cache.json`. Every subsequent run loads from cache — zero latency, fully reproducible.

**The cache is committed to the repository.** Reviewers reproduce results without an API key. Keywords absent from cache fall back to Component A only — graceful degradation, not a hard failure.

```json
{
  "margin_tier":           "premium | mid-range | commodity",
  "funnel_stage":          "awareness | consideration | decision",
  "competition_intensity": "low | medium | high"
}
```

**Why these three fields:**

- **`margin_tier`:** The optimal bid ceiling is a function of margin — a bid profitable for a premium product is a money-loser for a commodity. Tabular ROAS cannot separate margin from volume. Claude can.
- **`funnel_stage`:** Decision-stage keywords convert immediately; awareness-stage keywords require multiple exposures. Budget should concentrate where conversion is imminent.
- **`competition_intensity`:** Determines the bid *floor* needed to win auctions at all — not a score component. Feeds the bid floor modifier in Stage 5.

---

### Stage 4 — Unified Keyword Score

**Source:** All EDA findings converging

The score is a single continuous number summarising each keyword's expected ROI. It is the sole input to Stage 5.

**Behavioral component — weighted additive sum:**

```
behavioral_score = 0.60 × roas_vs_campaign_norm
                 + 0.25 × cvr_norm
                 + 0.15 × match_type_factor
```

Each term is **percentile-rank normalised** within the full dataset before blending, placing all factors on [0, 1]. Weights reflect the optimization target: ROAS dominates (0.60) because it is the metric being optimised; CVR (0.25) captures buyer quality independent of bid level; match type (0.15) acts as a prior on specificity.

**Why not CTR?** CTR is partially bid-driven — a higher bid wins better ad placement, which mechanically inflates CTR regardless of keyword quality. Including it in the score would reward keywords for spending more. Retained as a diagnostic column only.

**Why additive, not multiplicative?** Multiplication collapses to zero if any single term is weak — a keyword with zero conversions and good ROAS would score zero regardless of potential. The weighted sum ensures each signal contributes independently.

**Hybrid score formula:**

```
keyword_score = confidence × behavioral_score
              + (1 − confidence) × (0.60 × neighbor_roas_norm + 0.40 × llm_specificity)
              + llm_margin_modifier
              + llm_funnel_modifier
```

When `confidence` is low, the score falls back to a blend of two semantic signals: `neighbor_roas_norm` (actual earned ROAS of the 5 most similar keywords, 0.60 weight) and `llm_specificity` (how long-tail the keyword is, 0.40 weight). The KNN lookup is what makes the embedding layer genuinely load-bearing for sparse keywords.

**LLM modifier values:**

| Field | Value | Modifier |
|---|---|---|
| `margin_tier` | commodity | +0.00 |
| `margin_tier` | mid_range | +0.05 |
| `margin_tier` | premium | +0.10 |
| `funnel_stage` | awareness | +0.00 |
| `funnel_stage` | consideration | +0.05 |
| `funnel_stage` | decision | +0.10 |

`competition_intensity` is **not** a score additive — it feeds the bid floor modifier in Stage 5.

**The confidence weight is the load-bearing mechanism:**

| confidence | interpretation | score driven by |
|---|---|---|
| 1.0 | data-rich | 100% historical ROAS, CVR, match type |
| 0.5 | moderate data | 50% behavioral + 50% LLM specificity |
| 0.0 | zero history | 100% LLM semantic prior |

**Stability cap applied immediately after scoring:**

```
raw_bid = clip(raw_bid, current_avg_bid × 0.5, current_avg_bid × 1.5)
```

---

### Stage 5 — Asymmetric Portfolio Optimization

**Source:** EDA Q6 (all 10 campaigns 1.6×–4.8× over budget)

Symmetric scaling (divide all bids by the same factor) destroys the efficiency rank-ordering — the best and worst keywords are cut equally, wiping out everything Stages 1–4 computed.

**Step 1 — Score-to-bid derivation (budget satisfied by construction):**

```
expected_clicks = max(avg_daily_clicks, 0.5)   # floor prevents division by zero
raw_bid = (keyword_score / sum(campaign_keyword_scores)) × campaign_daily_budget / expected_clicks
```

`avg_daily_clicks` can be zero for keywords with impressions but no clicks. A floor of 0.5 avoids division-by-zero while keeping the budget allocation meaningful. These keywords are flagged in the reason column.

**Step 2 — Stability cap:**

```
raw_bid = clip(raw_bid, current_avg_bid × 0.50, current_avg_bid × 1.50)
```

Applied here — after bid derivation, not in Stage 4 where no bid exists yet. The floor modifier in Step 3 can still override this cap from below if needed.

**Step 3 — Competition bid floor modifier:**

```
floor_bid = max(0.20, 0.20 × (1 + 0.15 × competition_ordinal))
```

where `competition_ordinal`: low = 0, medium = 1, high = 2.
Floor values: low → $0.20, medium → $0.23, high → $0.26.

`competition_intensity` is not a score component — it sets the minimum bid needed to enter the auction, not the relative ranking of keywords.

**Step 4 — Asymmetric safety slash (if campaign still over budget after rounding):**

1. Compute `projected_spend = raw_bid × avg_daily_clicks` per keyword
2. If `sum(projected_spend) ≤ campaign_daily_budget` → no action needed
3. If over budget — asymmetric slash:
   - Sort keywords by `keyword_score` ascending (worst first)
   - Drive the lowest-score keyword's bid down toward its `floor_bid`
   - **`floor_bid` enforced inside the loop** — never go below on any iteration
   - Recompute projected spend after each reduction using the floored bid
   - Stop as soon as the campaign is within budget
   - **High-score keywords are never touched**

**Why the floor must be enforced inside the loop, not after:** if bids were allowed to go below `floor_bid` during optimization and then clipped back up at a later stage, those upward corrections would increase projected spend and re-violate the budget constraint the loop just satisfied.

**Edge case — budget floor exceeded:** if every keyword reaches its `floor_bid` and projected spend still exceeds the daily budget, the constraint cannot be satisfied — the campaign's minimum possible spend is above its budget. In this case: leave all keywords at `floor_bid` and flag the campaign in the reason column.

**Safety net:** if the slash loop completes but projected spend is still marginally over budget due to the floor constraint, apply proportional scale-down to the whole campaign, again enforcing `floor_bid` per keyword after scaling.

---

### Stage 6 — Marketplace Compliance & Output

- **Ceiling clip only:** `recommended_bid = min(bid, 15.00)` — the $0.20 floor is already guaranteed by Stage 5's loop. The only remaining compliance risk is bids above $15.00, which the stability cap (`current_avg_bid × 1.5`) can produce for high-bid keywords.
- Reason column: one sentence per keyword — key signals, budget action, and whether the score was LLM-primary
- Output: `keyword_id`, `keyword_text`, `current_avg_bid`, `recommended_bid`, `reason` → `bid_recommendations.csv`

> Full implementation with code and output samples: **Section 6** (Stages 1–2), **Section 7** (Stage 3), **Section 8** (Stages 4–5), **Section 9** (Stage 6) in [bid_optimizer.ipynb](bid_optimizer.ipynb)

---

### Complete Feature Plan

| Feature | Stage | Type | Formula / Source | EDA Anchor |
|---|---|---|---|---|
| `historical_roas` | 1 | Behavioral | total_revenue / total_spend | Q2 |
| `cvr` | 1 | Behavioral | total_conversions / total_clicks | Q2 |
| `ctr` | 1 | Diagnostic | total_clicks / total_impressions | Diagnostic only — bid-driven, excluded from score |
| `avg_daily_clicks` | 1 | Behavioral | total_clicks / days_active | Q1 |
| `avg_daily_spend` | 1 | Behavioral | total_spend / days_active | Q1 |
| `current_avg_bid` | 1 | Anchor | mean(current_bid) | Stability cap anchor |
| `match_type_factor` | 1 | Ordinal | exact=1.0, phrase=0.8, broad=0.5 | Q5 |
| `confidence` | 1 | Weight | sqrt(clicks/30 × days/7), capped at 1 | Q1, Q2 |
| `campaign_avg_roas` | 2 | Context | spend-weighted ROAS per campaign | Q4 |
| `roas_vs_campaign` | 2 | Relative | historical_roas / campaign_avg_roas | Q4 |
| `llm_specificity` | 3A | Semantic | cosine distance from embedding centroid | Long-tail vs generic |
| `neighbor_roas` | 3A | Semantic | similarity-weighted ROAS of k=5 nearest neighbors in embedding space | Cold-start prior from real historical data |
| `llm_margin_tier` | 3B | Semantic | Claude JSON → commodity=+0.00, mid_range=+0.05, premium=+0.10 | Margin blind spot |
| `llm_funnel_stage` | 3B | Semantic | Claude JSON → awareness=+0.00, consideration=+0.05, decision=+0.10 | Conversion proximity |
| `llm_competition` | 3B | Floor mod | Claude JSON → floor: low=$0.20, medium=$0.23, high=$0.26 | Bid floor only — not a score component |

---

## Operational Validation & Backtesting Framework

> Full implementation with charts and printed proofs: **Section 10** in [bid_optimizer.ipynb](bid_optimizer.ipynb)

### Why Not a Train/Test Split?

A temporal train/test split is the standard validation approach for forecasting models. This is not a forecasting model — it is a portfolio resource allocation optimizer. The distinction matters:

- A forecasting model generalises to unseen future data → held-out test set measures generalisation
- A portfolio optimizer uses all available signal to make one operational decision → there is no "generalisation" to measure, only constraint satisfaction and objective improvement

Splitting off days 22–30 as a test set would discard the 30% most recent and most relevant signal, producing a model optimised on stale data. The output quality is measured not by prediction accuracy but by two operational criteria: (1) are all constraints satisfied? (2) does the recommended allocation improve ROAS? Both are verifiable without a holdout set.

---

### Validation 1 — Hard Boundary Assertions

Programmatic `assert` checks prove 100% compliance with every hard constraint:

- **Marketplace bounds:** `min(recommended_bid) ≥ $0.20` and `max(recommended_bid) ≤ $15.00`
- **Stability cap:** no bid exceeds `current_avg_bid × 1.50` or falls below `current_avg_bid × 0.50` without justification

**Results:**
```
min recommended_bid = $0.20   (floor $0.20)     ✓
max recommended_bid = $4.09   (ceiling $15.00)  ✓
Upper-cap violations: 0                          ✓
Lower-cap violations (true): 0                   ✓
Floor-forced reductions: 101 keywords (budget constraint drove bids to marketplace floor — expected)
```

**Important nuance on the stability lower bound:** 101 keywords were driven to their marketplace floor (`$0.20–$0.26`) by the budget constraint. These sit below `current_avg_bid × 0.50` but are not violations — the budget constraint takes priority. Each is flagged in the reason column.

---

### Validation 2 — Budget Feasibility

Every campaign was 1.6×–4.8× over its daily budget in the historical data (EDA Q6). The optimized projection must bring all 10 campaigns within their caps.

A dual-bar chart shows all campaigns side-by-side:
- **Historical bar** — average actual daily spend (massively over budget)
- **Optimized bar** — `sum(recommended_bid × avg_daily_clicks)` per campaign
- **Red dashed line** — the hard daily budget cap

**Results — all 10 campaigns brought within budget:**

| Campaign | Historical spend | vs budget | Optimized spend | vs budget |
|---|---|---|---|---|
| Beauty & Personal Care | $571/day | 1.6× | $350/day | 1.00× ✓ |
| Electronics Deals | $836/day | 1.7× | $500/day | 1.00× ✓ |
| Fitness & Outdoors | $601/day | 3.0× | $200/day | 1.00× ✓ |
| Garden & Tools | $345/day | 4.3× | $80/day | 1.00× ✓ |
| Headphones & Audio | $684/day | 2.3× | $300/day | 1.00× ✓ |
| Home & Kitchen | $688/day | 1.7× | $400/day | 1.00× ✓ |
| Kids & Toys | $330/day | 2.2× | $150/day | 1.00× ✓ |
| Office Supplies | $471/day | 2.6× | $180/day | 1.00× ✓ |
| Pet Supplies | $572/day | 4.8× | $120/day | 1.00× ✓ |
| Phone Accessories | $748/day | 3.0× | $250/day | 1.00× ✓ |
| **Portfolio total** | **$5,845/day** | **2.3× avg** | **$2,529/day** | **1.00×** |

Total daily spend reduced from $5,845 to $2,529 — a 57% reduction — while concentrating the remaining budget on high-ROAS keywords.

---

### Validation 3 — Counterfactual ROAS Improvement

Since a live A/B test is not available, we use an offline economic proxy to prove the optimization improved efficiency:

**Simulation logic:**

```
revenue_per_click   = total_revenue / (total_clicks + 1e-5)
predicted_spend     = recommended_bid × avg_daily_clicks × 30
predicted_revenue   = predicted_spend × revenue_per_click × (recommended_bid / current_avg_bid)^0.1
```

The `(bid_new / bid_old)^0.1` term is a **diminishing-returns elasticity factor**. It makes the simulation conservative: raising a bid by 50% only increases revenue efficiency by ~4%, not 50%. This penalises aggressive upward moves and prevents the simulation from overstating the improvement.

**Results:**

```
Actual historical portfolio ROAS:    0.884x   (campaigns spending 1.6x–4.8x over budget)
Predicted optimized portfolio ROAS:  2.218x
Improvement:                        +150.9%
```

**Per-campaign ROAS improvement:**

| Campaign | Actual ROAS | Predicted ROAS | Change |
|---|---|---|---|
| Fitness & Outdoors | 0.77× | 2.31× | +198% |
| Electronics Deals | 0.95× | 2.59× | +173% |
| Office Supplies | 0.64× | 1.69× | +165% |
| Phone Accessories | 0.82× | 2.10× | +158% |
| Beauty & Personal Care | 0.87× | 2.22× | +154% |
| Headphones & Audio | 0.95× | 2.37× | +149% |
| Garden & Tools | 0.94× | 2.26× | +141% |
| Kids & Toys | 1.26× | 2.78× | +120% |
| Home & Kitchen | 0.80× | 1.75× | +119% |
| Pet Supplies | 0.98× | 2.01× | +105% |

ROAS improves across all 10 campaigns — no campaign is sacrificed to boost another.

**Why this is a strong proof:** the historical ROAS baseline (0.88×) was computed from campaigns spending 1.6×–4.8× their daily budgets — every keyword running unconstrained, money-losing ones included. The algorithm achieves 2.22× ROAS while spending *within* budget. Better efficiency and budget compliance simultaneously, not a trade-off.

**How the improvement happens:** the algorithm stops spending money on keywords that lose money (cuts their bids to the $0.20 floor) and gives that budget to keywords that are already profitable. Spending less in total but only on the right keywords produces a higher return per dollar — which is exactly what ROAS measures.

**Why this is not data leakage.** Traditional data leakage means using information at model-build time that wouldn't exist at prediction time. That is not the case here — we use 30 days of historical data to produce a recommendation for the *next* day; no future information touches the scoring or bid derivation. The offline simulation then asks a counterfactual question ("what would ROAS have been if these bids had been used over the past 30 days?") using the same window. This is the standard validation approach for operational bid optimizers where no live holdout is available. The improvement looks large because the baseline was computed from campaigns spending 1.6×–4.8× over budget — the starting point was artificially bad. The real test is a live A/B holdout, described in Part B.

**Known limitations of this proxy:**
- Revenue-per-click is assumed constant — real auctions are dynamic and competitor bids react
- The elasticity exponent (0.1) is a conservative approximation, not a fitted bid-response curve (fitting a curve requires much more bid variation than 3–5 values per keyword — see Section 5.0)
- This is an offline simulation, not a controlled experiment; causal claims require live testing

These limitations are inherent to any offline bid optimizer. The conservative elasticity factor is chosen specifically to avoid over-claiming — if the simulation shows ROAS improvement under a conservative penalty, the real improvement is likely at least as large.

---

## Part B — Reasoning About Uncertainty

**Sparse data.** The confidence score — `sqrt(min(clicks/30,1) × min(days/7,1))` — gates how much behavioral vs. semantic signal drives the keyword's bid. At `confidence = 0`, the score is 100% semantic: a KNN prior over the k=5 most similar keywords in embedding space (similarity-weighted average of their actual historical ROAS) plus an LLM specificity signal from the keyword's distance from the portfolio centroid. Claude's margin and funnel modifiers apply regardless of confidence level. The ±50% stability cap ensures a miscalibrated prior causes a modest one-cycle over-bid, not a budget event — and as data accumulates, the weight shifts automatically back to behavioral signal. No special-case logic anywhere.

**Production failure modes.** (1) *Feedback loops:* bid raised → noisy week with no conversions → score falls → bid cut → less traffic → score falls further. The ±50% cap slows oscillation; a production system should also hold bids steady after two consecutive direction reversals on the same keyword. (2) *Data pipeline corruption:* duplicate rows inflate ROAS; a tracking outage zeroes revenue for two days, triggering aggressive cuts to keywords that actually earned. Mitigation: pre-cycle validation on row count, date coverage, and revenue/spend ratio bounds — hold all bids and alert on failure. (3) *Budget infeasibility:* if a campaign's minimum-floor spend (n keywords × $0.20 floor × avg clicks) exceeds its daily budget, no valid bid set exists. The algorithm sets all bids to floor and flags it, but production needs a human alert — this is a budget configuration problem, not an optimizer problem. (4) *Seasonality:* the 30-day flat aggregation is blind to seasonal trends. December holiday ROAS carried into January over-bids seasonal keywords and under-bids evergreen ones. A production rollout would add a recency-weighted ROAS window.

**Measuring success.** Primary metric: portfolio ROAS (revenue / spend) vs. a held-out control group of campaigns on manual bidding. Without a holdout, any ROAS change is confounded by seasonality and competitor activity. Timeline: week 1 — do not report ROAS, Amazon's attribution window is up to 14 days so early numbers are structurally understated; week 2 — first attributable read, verify budget compliance before interpreting efficiency numbers; weeks 3–4 — first statistically meaningful ROAS comparison vs. holdout. Secondary signals: budget utilisation rate per campaign, bid oscillation rate, and floor-hit rate. The offline simulation showed 0.884× → 2.218× ROAS under conservative elasticity assumptions; a live target of ≥+30% lift over the holdout by week 4 with zero budget violations is a sensible first deployment gate.

> Implementation details and formula derivations: **Section 8** in [bid_optimizer.ipynb](bid_optimizer.ipynb)

---

## Part C — Cold-Start for New Keywords

**How the algorithm handles zero-history keywords.** A new campaign with 50 keywords and no click history has `confidence = 0` for every keyword, so the hybrid score reduces to:

```
keyword_score = 0.60 × neighbor_roas_norm + 0.40 × llm_specificity
              + llm_margin_modifier + llm_funnel_modifier
```

`neighbor_roas_norm` is the similarity-weighted average ROAS of the k=5 most semantically similar keywords *already in the portfolio* — so even brand-new keywords get a cold-start prior grounded in real campaign data, not world-knowledge alone. `llm_specificity` (centroid distance in embedding space) differentiates long-tail high-intent queries from generic head terms. Claude's margin and funnel modifiers add commercial context. The result: "premium noise cancelling headphones men's" starts with a meaningfully higher score than "what is bluetooth" before a single click is recorded.

**The exploration/efficiency tension.** The natural outcome of the above is that all 50 new keywords receive bids derived from moderate, undifferentiated priors — and moderate bids on an unknown keyword produce few impressions, which means confidence stays near zero for many cycles. The system is efficient (it doesn't waste budget on keywords the prior scores poorly) but slow to learn: a keyword the prior underestimates may never receive enough traffic to prove its actual value.

The correct response is an explicit exploration budget for the first phase. For cycles 1–7, reserve 20–25% of the campaign's daily budget as a uniform exploration allocation spread evenly across all keywords, regardless of score. This guarantees every keyword reaches at least a minimum impression floor. The remaining 75–80% follows the score-weighted allocation as normal. After 7 days, most keywords have accumulated enough clicks and active days for `confidence` to become meaningful, and the exploration allocation is retired — the optimizer takes over fully with real behavioral signal.

This is deliberately simple: no Thompson sampling, no bandit arms. The reason is that the confidence score already encodes the exploration/exploitation transition continuously. The flat exploration slice just ensures the transition happens within a week rather than over many cycles, at a known, bounded cost (25% of budget × 7 days).

**Expected trajectory.** Days 1–7: uniform exploration, KNN/LLM priors dominate score. Days 8–14: behavioral signal starts mixing in for keywords with sufficient traffic; worst performers begin receiving less budget. Week 3+: high-confidence keywords drive the allocation; the cold-start phase is effectively over and the system behaves identically to an established campaign.

> Score formula and confidence mechanics: **Section 8** in [bid_optimizer.ipynb](bid_optimizer.ipynb)

