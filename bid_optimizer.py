"""
Core bid optimizer module — imported by main.py and used in the notebook.
Full explanation and reasoning live in bid_optimizer.ipynb.
"""
import pandas as pd
import numpy as np


class BidOptimizer:
    BID_MIN = 0.20
    BID_MAX = 15.00
    MIN_CLICKS_FOR_FULL_CONFIDENCE = 30
    MIN_DAYS_FOR_FULL_CONFIDENCE = 7
    MAX_BID_CHANGE_FACTOR = 2.0
    MIN_BID_CHANGE_FACTOR = 0.5

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])

    def _aggregate(self) -> pd.DataFrame:
        kw = self.df.groupby(
            ['keyword_id', 'keyword_text', 'campaign_id', 'match_type']
        ).agg(
            total_spend=('spend', 'sum'),
            total_revenue=('revenue', 'sum'),
            total_clicks=('clicks', 'sum'),
            total_conversions=('conversions', 'sum'),
            days_active=('date', 'nunique'),
            current_bid=('current_bid', 'last'),
            campaign_daily_budget=('campaign_daily_budget', 'last')
        ).reset_index()

        kw['roas'] = np.where(
            kw['total_spend'] > 0,
            kw['total_revenue'] / kw['total_spend'],
            np.nan
        )
        kw['avg_daily_clicks'] = kw['total_clicks'] / kw['days_active'].clip(lower=1)
        return kw

    def _compute_confidence(self, kw: pd.DataFrame) -> pd.Series:
        click_conf = (kw['total_clicks'] / self.MIN_CLICKS_FOR_FULL_CONFIDENCE).clip(0, 1)
        day_conf = (kw['days_active'] / self.MIN_DAYS_FOR_FULL_CONFIDENCE).clip(0, 1)
        return np.sqrt(click_conf * day_conf)

    def _compute_target_roas(self, kw: pd.DataFrame) -> pd.Series:
        camp_roas = kw.groupby('campaign_id').apply(
            lambda g: g['total_revenue'].sum() / g['total_spend'].sum()
            if g['total_spend'].sum() > 0 else 1.0
        ).rename('target_roas')
        return kw['campaign_id'].map(camp_roas)

    def _apply_budget_constraint(self, kw: pd.DataFrame) -> pd.DataFrame:
        kw = kw.copy()
        kw['projected_daily_spend'] = kw['recommended_bid'] * kw['avg_daily_clicks']
        camp_projected = kw.groupby('campaign_id')['projected_daily_spend'].transform('sum')
        budget_factor = np.where(
            camp_projected > kw['campaign_daily_budget'],
            kw['campaign_daily_budget'] / camp_projected,
            1.0
        )
        kw['recommended_bid'] = (kw['recommended_bid'] * budget_factor).clip(
            self.BID_MIN, self.BID_MAX
        ).round(2)
        return kw

    def optimize(self) -> pd.DataFrame:
        kw = self._aggregate()
        kw['confidence'] = self._compute_confidence(kw)
        kw['target_roas'] = self._compute_target_roas(kw)

        roas_ratio = (kw['roas'] / kw['target_roas'].replace(0, np.nan)).clip(
            self.MIN_BID_CHANGE_FACTOR, self.MAX_BID_CHANGE_FACTOR
        )

        scaled_bid = kw['current_bid'] * roas_ratio
        kw['recommended_bid'] = (
            kw['confidence'] * scaled_bid + (1 - kw['confidence']) * kw['current_bid']
        ).fillna(kw['current_bid']).clip(self.BID_MIN, self.BID_MAX).round(2)

        kw = self._apply_budget_constraint(kw)

        return kw[[
            'keyword_id', 'keyword_text', 'campaign_id', 'match_type',
            'current_bid', 'recommended_bid', 'roas', 'target_roas',
            'confidence', 'total_clicks', 'days_active'
        ]]
