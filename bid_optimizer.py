"""
Keyword Bid Optimization Engine — core module.

Full explanation of every design decision, formula, and trade-off lives in
bid_optimizer.ipynb (Sections 5–9). This module is the production-ready
implementation of that notebook's algorithm.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

BID_MIN        = 0.20
BID_MAX        = 15.00
STABILITY_LOW  = 0.50
STABILITY_HIGH = 1.50
KNN_K          = 5
CACHE_PATH     = Path("keyword_metadata_cache.json")

_MARGIN_MODS = {"commodity": 0.00, "mid-range": 0.05, "premium": 0.10}
_FUNNEL_MODS = {"awareness": 0.00, "consideration": 0.05, "decision": 0.10}
_COMP_FLOORS = {"low": 0.20, "medium": 0.23, "high": 0.26}
_FALLBACK    = {"margin_tier": "mid-range", "funnel_stage": "consideration",
                "competition_intensity": "medium"}
_MARGIN_LABELS = {0.00: "commodity", 0.05: "mid-range", 0.10: "premium"}
_FUNNEL_LABELS = {0.00: "awareness", 0.05: "consideration", 0.10: "decision"}

_PROMPT = (
    'Classify this Amazon advertising keyword for bid optimization.\n'
    'Keyword: "{kw}"\n\n'
    'Return ONLY a valid JSON object with exactly these fields:\n'
    '{{"margin_tier": "premium|mid-range|commodity", '
    '"funnel_stage": "awareness|consideration|decision", '
    '"competition_intensity": "low|medium|high"}}'
)


def _pct_rank(series: pd.Series) -> pd.Series:
    r = rankdata(series.fillna(0).values, method="average")
    return pd.Series((r - 1) / max(len(r) - 1, 1), index=series.index)


class BidOptimizer:
    """
    Portfolio bid optimizer: scores every keyword by efficiency, derives bids
    from budget-proportional allocation, and enforces all marketplace constraints.

    Stages (see bid_optimizer.ipynb §5 for full design rationale):
      1. Behavioral feature engineering (aggregate 30-day log → keyword profiles)
      2. Campaign contextualization (relative ROAS within budget pool)
      3. LLM semantic layer (embeddings + Claude metadata)
      4. Unified keyword score (confidence-weighted hybrid)
      5. Asymmetric portfolio optimization (score-to-bid + safety slash)
      6. Marketplace compliance & output
    """

    def __init__(self, df: pd.DataFrame, api_key: str | None = None):
        self.df      = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._cache: dict = (
            json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
        )

    # ── Stage 1 & 2: Feature Engineering ─────────────────────────────────────

    def _build_profiles(self) -> pd.DataFrame:
        kw = self.df.groupby(
            ["keyword_id", "keyword_text", "campaign_id", "campaign_name", "match_type"]
        ).agg(
            total_spend           = ("spend",                "sum"),
            total_revenue         = ("revenue",              "sum"),
            total_clicks          = ("clicks",               "sum"),
            total_impressions     = ("impressions",          "sum"),
            total_conversions     = ("conversions",          "sum"),
            days_active           = ("date",                 "nunique"),
            current_avg_bid       = ("current_bid",          "mean"),
            campaign_daily_budget = ("campaign_daily_budget","last"),
        ).reset_index()

        kw["avg_daily_clicks"] = kw["total_clicks"] / kw["days_active"]
        kw["avg_daily_spend"]  = kw["total_spend"]  / kw["days_active"]

        kw["historical_roas"] = np.where(
            kw["total_spend"] > 0, kw["total_revenue"] / kw["total_spend"], 0.0)
        kw["cvr"] = np.where(
            kw["total_clicks"] > 0, kw["total_conversions"] / kw["total_clicks"], 0.0)

        kw["match_type_factor"] = kw["match_type"].map(
            {"exact": 1.0, "phrase": 0.8, "broad": 0.5})

        click_sat       = (kw["total_clicks"] / 30).clip(0, 1)
        day_sat         = (kw["days_active"]  /  7).clip(0, 1)
        kw["confidence"] = np.sqrt(click_sat * day_sat)

        camp = kw.groupby("campaign_id").agg(
            _r=("total_revenue","sum"), _s=("total_spend","sum")).reset_index()
        camp["campaign_avg_roas"] = np.where(
            camp["_s"] > 0, camp["_r"] / camp["_s"], 0.0)
        kw = kw.merge(camp[["campaign_id","campaign_avg_roas"]], on="campaign_id")
        kw["roas_vs_campaign"] = np.where(
            kw["campaign_avg_roas"] > 0,
            kw["historical_roas"] / kw["campaign_avg_roas"], 0.0)

        return kw

    # ── Stage 3: LLM Semantic Layer ───────────────────────────────────────────

    def _embed(self, kw: pd.DataFrame) -> pd.DataFrame:
        embedder   = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = np.array(embedder.encode(
            kw["keyword_text"].tolist(), show_progress_bar=False, batch_size=32))

        centroid             = embeddings.mean(axis=0, keepdims=True)
        kw["llm_specificity"] = 1.0 - cos_sim(embeddings, centroid).flatten()

        sim_matrix = cos_sim(embeddings)
        np.fill_diagonal(sim_matrix, 0.0)
        roas_vals = kw["historical_roas"].values
        neighbor_roas = []
        for i in range(len(kw)):
            top_k   = np.argsort(sim_matrix[i])[-KNN_K:]
            weights = sim_matrix[i][top_k]
            nr = ((weights * roas_vals[top_k]).sum() / weights.sum()
                  if weights.sum() > 0 else roas_vals.mean())
            neighbor_roas.append(nr)
        kw["neighbor_roas"] = neighbor_roas

        return kw

    def _fetch_claude_metadata(self, kw_text: str) -> dict:
        if kw_text in self._cache:
            return self._cache[kw_text]
        if not self.api_key:
            self._cache[kw_text] = _FALLBACK
            return _FALLBACK
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            resp   = client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=120,
                messages=[{"role":"user",
                           "content": _PROMPT.format(kw=kw_text)}])
            result = json.loads(resp.content[0].text.strip())
            assert all(k in result for k in _FALLBACK)
            self._cache[kw_text] = result
        except Exception:
            self._cache[kw_text] = _FALLBACK
        return self._cache[kw_text]

    def _apply_llm_metadata(self, kw: pd.DataFrame) -> pd.DataFrame:
        missing = [t for t in kw["keyword_text"].unique() if t not in self._cache]
        for t in missing:
            self._fetch_claude_metadata(t)
        if missing:
            CACHE_PATH.write_text(json.dumps(self._cache, indent=2))

        kw["llm_margin_mod"] = kw["keyword_text"].map(
            lambda t: _MARGIN_MODS.get(self._cache.get(t, _FALLBACK)["margin_tier"], 0.05))
        kw["llm_funnel_mod"] = kw["keyword_text"].map(
            lambda t: _FUNNEL_MODS.get(self._cache.get(t, _FALLBACK)["funnel_stage"], 0.05))
        kw["floor_bid_comp"] = kw["keyword_text"].map(
            lambda t: _COMP_FLOORS.get(
                self._cache.get(t, _FALLBACK)["competition_intensity"], 0.23))
        return kw

    # ── Stage 4: Unified Keyword Score ────────────────────────────────────────

    def _score(self, kw: pd.DataFrame) -> pd.DataFrame:
        kw["roas_vs_campaign_norm"] = _pct_rank(kw["roas_vs_campaign"])
        kw["cvr_norm"]              = _pct_rank(kw["cvr"])
        kw["neighbor_roas_norm"]    = _pct_rank(kw["neighbor_roas"])

        kw["behavioral_score"] = (
            0.60 * kw["roas_vs_campaign_norm"] +
            0.25 * kw["cvr_norm"] +
            0.15 * kw["match_type_factor"]
        )
        kw["semantic_prior"] = (
            0.60 * kw["neighbor_roas_norm"] +
            0.40 * kw["llm_specificity"]
        )
        kw["keyword_score"] = (
            kw["confidence"]       * kw["behavioral_score"] +
            (1 - kw["confidence"]) * kw["semantic_prior"] +
            kw["llm_margin_mod"] +
            kw["llm_funnel_mod"]
        )
        return kw

    # ── Stage 5: Asymmetric Portfolio Optimization ────────────────────────────

    def _optimize_campaign(self, camp_df: pd.DataFrame) -> pd.DataFrame:
        df     = camp_df.copy()
        budget = df["campaign_daily_budget"].iloc[0]

        df["expected_clicks"] = df["avg_daily_clicks"].clip(lower=0.5)
        total_score = df["keyword_score"].sum()
        df["raw_bid"] = (
            (df["keyword_score"] / total_score) * budget / df["expected_clicks"]
        )

        df["raw_bid"] = df["raw_bid"].clip(
            lower=df["current_avg_bid"] * STABILITY_LOW,
            upper=df["current_avg_bid"] * STABILITY_HIGH,
        )

        df["floor_bid"] = df["floor_bid_comp"].clip(lower=BID_MIN)
        df["raw_bid"]   = df[["raw_bid","floor_bid"]].max(axis=1)

        df["projected_spend"] = df["raw_bid"] * df["expected_clicks"]

        if df["projected_spend"].sum() > budget + 0.01:
            df = df.sort_values("keyword_score").copy()
            for idx in df.index:
                if df["projected_spend"].sum() <= budget + 0.01:
                    break
                excess   = df["projected_spend"].sum() - budget
                bid      = df.loc[idx, "raw_bid"]
                floor    = df.loc[idx, "floor_bid"]
                clicks   = df.loc[idx, "expected_clicks"]
                headroom = (bid - floor) * clicks
                if headroom <= 0:
                    continue
                cut = min(excess, headroom)
                df.loc[idx, "raw_bid"]         = bid - cut / clicks
                df.loc[idx, "projected_spend"] = df.loc[idx, "raw_bid"] * clicks

            if df["projected_spend"].sum() > budget + 0.01:
                scale         = budget / df["projected_spend"].sum()
                df["raw_bid"] = (df["raw_bid"] * scale).clip(lower=df["floor_bid"])
                df["projected_spend"] = df["raw_bid"] * df["expected_clicks"]

        return df

    # ── Stage 6: Compliance & Output ─────────────────────────────────────────

    def _build_reason(self, row: pd.Series) -> str:
        parts = []
        conf  = row["confidence"]
        if conf >= 0.8:
            parts.append(
                f"behavioral: ROAS {row['historical_roas']:.2f}x "
                f"(campaign avg {row['campaign_avg_roas']:.2f}x, "
                f"relative {row['roas_vs_campaign']:.2f}x)")
        elif conf <= 0.2:
            parts.append(
                f"LLM-primary (confidence={conf:.2f}): "
                f"neighbor ROAS prior {row['neighbor_roas']:.2f}x")
        else:
            parts.append(
                f"blended (confidence={conf:.2f}): "
                f"ROAS {row['historical_roas']:.2f}x + "
                f"neighbor prior {row['neighbor_roas']:.2f}x")

        parts.append(
            f"margin={_MARGIN_LABELS.get(row['llm_margin_mod'],'mid-range')}, "
            f"funnel={_FUNNEL_LABELS.get(row['llm_funnel_mod'],'consideration')}")

        change_pct = (row["recommended_bid"] / row["current_avg_bid"] - 1) * 100
        if row["recommended_bid"] <= row["floor_bid"] + 0.01:
            parts.append("bid at marketplace floor (budget constraint)")
        elif change_pct >= 40:
            parts.append(f"bid raised +{change_pct:.0f}% (high efficiency)")
        elif change_pct <= -40:
            parts.append(f"bid cut {change_pct:.0f}% (low efficiency)")

        return "; ".join(parts)

    # ── Public API ────────────────────────────────────────────────────────────

    def optimize(self) -> pd.DataFrame:
        kw = self._build_profiles()
        kw = self._embed(kw)
        kw = self._apply_llm_metadata(kw)
        kw = self._score(kw)

        results = [self._optimize_campaign(g) for _, g in kw.groupby("campaign_id")]
        kw_opt  = pd.concat(results).reset_index(drop=True)

        kw_opt["recommended_bid"] = kw_opt["raw_bid"].clip(upper=BID_MAX).round(2)
        kw_opt["reason_or_score"] = kw_opt.apply(self._build_reason, axis=1)

        return kw_opt[[
            "keyword_id", "keyword_text", "current_avg_bid",
            "recommended_bid", "reason_or_score",
        ]]
