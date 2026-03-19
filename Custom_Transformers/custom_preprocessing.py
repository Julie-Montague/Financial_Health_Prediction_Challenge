import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import random
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

class FHIDataCleaner(BaseEstimator, TransformerMixin):
    """

    Key choices:
    - raw_missing_count = original missingness before any cleaning
    - clean_missing_count = missingness after numeric corrections, before final cat fill
    - zeros are preserved and flagged, not blanket-converted to NaN
    - country-aware winsorization is fit on training data only
    - categorical NaNs are filled with "missing" at the end so downstream string logic works
    """

    def __init__(
        self,
        winsor_quantile=0.99,
        burn_flag_threshold=10.0,
        burn_null_threshold=50.0,
        min_owner_start_age=18,
    ):
        self.winsor_quantile = winsor_quantile
        self.burn_flag_threshold = burn_flag_threshold
        self.burn_null_threshold = burn_null_threshold
        self.min_owner_start_age = min_owner_start_age

        self.sensitive = ["personal_income", "business_expenses", "business_turnover"]
        self.insurance_cols = [
            "motor_vehicle_insurance",
            "medical_insurance",
            "funeral_insurance",
        ]
        self.age_cols = ["owner_age", "business_age"]
        self.skip_text_cols = {"ID", "Target", "stratify_col"}

    def fit(self, X, y=None):
        out = X.copy()

        out = self._normalize_text(out)
        out = self._coerce_numeric(out)
        out = self._apply_logical_numeric_rules(out)

        self.global_caps_ = {}
        self.country_caps_ = {}

        for col in self.sensitive:
            if col not in out.columns:
                continue

            s = pd.to_numeric(out[col], errors="coerce")
            if s.notna().any():
                self.global_caps_[col] = float(s.quantile(self.winsor_quantile))
            else:
                self.global_caps_[col] = np.nan

            if "country" in out.columns:
                caps = out.groupby("country", dropna=False)[col].quantile(self.winsor_quantile)
                self.country_caps_[col] = caps.to_dict()
            else:
                self.country_caps_[col] = {}

        return self

    def transform(self, X):
        out = X.copy()

        # 1) original missingness before any cleaning
        out["raw_missing_count"] = out.isna().sum(axis=1)

        # 2) normalize text
        out = self._normalize_text(out)

        # 3) numeric coercion
        out = self._coerce_numeric(out)

        # 4) logical numeric fixes
        out = self._apply_logical_numeric_rules(out)

        # 5) stable train-defined winsorization
        out = self._apply_winsorization(out)

        # 6) logical insurance backfill (before final cat fill)
        out = self._apply_insurance_logic(out)

        # 7) final numeric flags after all numeric corrections
        for col in self.sensitive:
            if col in out.columns:
                s = pd.to_numeric(out[col], errors="coerce")
                out[f"{col}_is_zero"] = s.eq(0).astype(int)
                out[f"{col}_is_missing"] = s.isna().astype(int)

        # 8) missing count after numeric rules, before final categorical fill
        out["clean_missing_count"] = out.isna().sum(axis=1)

        # 9) final categorical fill for downstream feature engineering
        cat_cols_to_fill = [
            c
            for c in out.select_dtypes(include=["object", "category", "string"]).columns
            if c not in self.skip_text_cols
        ]
        for c in cat_cols_to_fill:
            out[c] = out[c].fillna("missing")

        return out

    # -------------------------
    # Helper methods
    # -------------------------
    def _normalize_text(self, out):
        cat_cols = [
            c
            for c in out.select_dtypes(include=["object", "category", "string"]).columns
            if c not in self.skip_text_cols
        ]

        bad_missing = {
            "dont know or n/a",
            "dont know",
            "dont know / doesnt apply",
            "do not know / n/a",
            "dont know (do not show)",
            "refused",
            "",
        }

        typo_fixes = {
            "0": "no",
            "1": "yes",
        }

        for col in cat_cols:
            s = out[col].astype("string")

            s = s.str.strip()
            s = s.str.replace(r"\s+", " ", regex=True)
            s = s.str.replace(r"['’`´?\u200e]", "", regex=True)
            s = s.str.lower()

            s = s.replace(list(bad_missing), pd.NA)
            s = s.replace(typo_fixes)

            out[col] = s

        return out

    def _coerce_numeric(self, out):
        numeric_cols = [c for c in (self.sensitive + self.age_cols) if c in out.columns]
        for col in numeric_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out

    def _apply_logical_numeric_rules(self, out):
        # Business age cannot be less than zero
        for col in ["owner_age", "business_age"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
                out.loc[out[col] < 0, col] = np.nan

        # Flag: business appears older than current owner could have personally started
        if "business_age" in out.columns and "owner_age" in out.columns:
            plausible_self_start_years = (out["owner_age"] - self.min_owner_start_age).clip(lower=0)
            out["business_older_than_owner_career"] = (
                out["business_age"] > plausible_self_start_years
            ).fillna(False).astype(int)

        # Burn-rate feature: flag high-burn cases, only null out truly absurd ones
        if "business_expenses" in out.columns and "business_turnover" in out.columns:
            be = pd.to_numeric(out["business_expenses"], errors="coerce")
            bt = pd.to_numeric(out["business_turnover"], errors="coerce")
            burn_ratio = be / (bt + 1.0)

            out["extreme_burn_flag"] = burn_ratio.gt(self.burn_flag_threshold).fillna(False).astype(int)

            # Only set to NaN for truly implausible cases
            absurd_mask = burn_ratio.gt(self.burn_null_threshold).fillna(False)
            out.loc[absurd_mask, "business_expenses"] = np.nan

        return out

    #outlier capping
    def _apply_winsorization(self, out):
        if not hasattr(self, "global_caps_"):
            return out

        for col in self.sensitive:
            if col not in out.columns:
                continue

            s = pd.to_numeric(out[col], errors="coerce")

            if "country" in out.columns and hasattr(self, "country_caps_"):
                country_cap_map = self.country_caps_.get(col, {})
                caps = out["country"].map(country_cap_map)

                global_cap = self.global_caps_.get(col, np.nan)
                if not pd.isna(global_cap):
                    caps = caps.fillna(global_cap)

                valid_cap = caps.notna()
                s = s.where(~valid_cap, np.minimum(s, caps))
            else:
                global_cap = self.global_caps_.get(col, np.nan)
                if not pd.isna(global_cap):
                    s = s.clip(upper=global_cap)

            out[col] = s

        return out

    def _apply_insurance_logic(self, out):
        if "has_insurance" in out.columns:
            has_ins = out["has_insurance"].astype("string")

            for ins in self.insurance_cols:
                if ins in out.columns:
                    missing_ins = out[ins].isna()
                    out.loc[(has_ins == "no") & missing_ins, ins] = "never had"

        return out
    

    
