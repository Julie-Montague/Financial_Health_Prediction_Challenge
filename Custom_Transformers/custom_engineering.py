import pandas as pd
import numpy as np
from collections import Counter
import re
from sklearn.cluster import KMeans
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold

import random
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

class FHIBaseSignals(BaseEstimator, TransformerMixin):
    """Row-level domain logic, textual mapping, and basic ratios."""

    def fit(self, X, y=None):
        self._fit_missingness_reference(X)
        return self

    def transform(self, X):
        out = X.copy()
        eps = 1e-6

        # 1) Relative missingness
        if "country" in out.columns and "raw_missing_count" in out.columns and hasattr(self, "country_missing_means_"):
            c_means = out["country"].map(self.country_missing_means_).fillna(getattr(self, "global_missing_mean_", 0.0))
            out["relative_missingness"] = out["raw_missing_count"] - c_means

        # 2) Formalization ordinals
        record_map = {"no": 0.0, "yes, sometimes": 1.0, "yes": 2.0, "yes, always": 3.0, "missing": np.nan}
        tax_map = {"no": 0.0, "dont know": 0.0, "refused": 0.0, "yes": 1.0, "missing": np.nan}
        mm_map = {"never had": 0.0, "used to have but dont have now": 0.5, "have now": 1.0, "missing": np.nan}

        form_cols = []
        if "keeps_financial_records" in out.columns:
            out["records_ordinal"] = self._norm_str(out["keeps_financial_records"]).map(record_map)
            form_cols.append("records_ordinal")
        if "compliance_income_tax" in out.columns:
            out["tax_ordinal"] = self._norm_str(out["compliance_income_tax"]).map(tax_map)
            form_cols.append("tax_ordinal")
        if "has_mobile_money" in out.columns:
            out["mobile_money_ordinal"] = self._norm_str(out["has_mobile_money"]).map(mm_map)
            form_cols.append("mobile_money_ordinal")

        out["master_formalization_index"] = out[form_cols].mean(axis=1) if form_cols else np.nan

        # 3) Access blocks
        digital_cols = ["has_mobile_money", "has_internet_banking", "has_cellphone"]
        finance_cols = ["has_debit_card", "has_credit_card", "has_loan_account"]
        insurance_cols = ["has_insurance", "medical_insurance", "funeral_insurance", "motor_vehicle_insurance"]

        out["digital_access_score"] = self._block_current_score(out, digital_cols)
        out["formal_finance_score"] = self._block_current_score(out, finance_cols)
        out["insurance_access_score"] = self._block_current_score(out, insurance_cols)

        inc_blocks = [c for c in ["digital_access_score", "formal_finance_score", "insurance_access_score"] if c in out.columns]
        out["digital_inclusion_score"] = out[inc_blocks].mean(axis=1) if inc_blocks else np.nan

        out["lost_financial_product_count"] = self._count_lost_products(out, ["has_mobile_money", "has_internet_banking", "has_debit_card", "has_credit_card", "has_loan_account", "medical_insurance", "funeral_insurance", "motor_vehicle_insurance"])

        # 3b) Access loss / mismatch features
        out["digital_formal_gap"] = out["digital_access_score"].fillna(0) - out["formal_finance_score"].fillna(0)

        tmp_funding = pd.DataFrame(index=out.index)
        for col in ["uses_friends_family_savings", "uses_informal_lender"]:
            if col in out.columns:
                s = self._norm_str(out[col])
                tmp_funding[f"{col}_current"] = self._is_current_access(s).astype(float)
                tmp_funding[f"{col}_lost"] = self._is_lost_access(s).astype(float)

        if tmp_funding.shape[1] > 0:
            c_cols = [c for c in tmp_funding.columns if c.endswith("_current")]
            l_cols = [c for c in tmp_funding.columns if c.endswith("_lost")]
            out["informal_funding_reliance"] = tmp_funding[c_cols].sum(axis=1) if c_cols else 0.0
            out["informal_funding_loss"] = tmp_funding[l_cols].sum(axis=1) if l_cols else 0.0
        else:
            out["informal_funding_reliance"] = np.nan
            out["informal_funding_loss"] = np.nan

        if "perception_insurance_important" in out.columns and "has_insurance" in out.columns:
            ins_imp = self._is_current_yes(self._norm_str(out["perception_insurance_important"])).astype(float)
            has_ins = self._is_current_access(self._norm_str(out["has_insurance"])).astype(float)
            out["insurance_need_gap"] = ins_imp * (1.0 - has_ins)
        else:
            out["insurance_need_gap"] = np.nan

        ins_parts = [self._is_current_yes(self._norm_str(out[c])).astype(float) for c in ["perception_cannot_afford_insurance", "perception_insurance_doesnt_cover_losses", "perception_insurance_companies_dont_insure_businesses_like_yours"] if c in out.columns]
        out["insurance_constraint_index"] = pd.concat(ins_parts, axis=1).sum(axis=1) if ins_parts else np.nan

        # 4) Missingness by block
        out["n_missing_finance_access"] = self._count_missing(out, digital_cols + finance_cols)
        out["n_missing_insurance"] = self._count_missing(out, insurance_cols)
        out["n_missing_formality"] = self._count_missing(out, ["keeps_financial_records", "compliance_income_tax", "has_mobile_money"])
        out["n_missing_attitudes"] = self._count_missing(out, ["attitude_stable_business_environment", "attitude_satisfied_with_achievement", "attitude_more_successful_next_year", "attitude_worried_shutdown", "current_problem_cash_flow", "problem_sourcing_money"])

        # 5) Money logs and ratios
        bt = self._num_series(out, "business_turnover")
        be = self._num_series(out, "business_expenses")
        pi = self._num_series(out, "personal_income")
        bt_safe = np.maximum(bt, 0) + eps

        out["burn_rate"] = be / bt_safe
        out["owner_extraction_rate"] = np.clip(np.maximum(pi, 0) / bt_safe, a_min=None, a_max=2.0)
        out["personal_wealth_buffer"] = np.maximum(pi, 0) / (np.maximum(bt, 0) + 1.0)
        out["wealth_opacity_ratio"] = np.maximum(bt, 0) / (self._num_series(out, "raw_missing_count") + 1.0) if "raw_missing_count" in out.columns else np.nan

        for col in ["personal_income", "business_expenses", "business_turnover"]:
            if col in out.columns: out[f"{col}_log"] = np.log1p(self._num_series(out, col).clip(lower=0))

        # 6) Business maturity
        b_age_col = self._get_business_age_col(out)
        if b_age_col is not None:
            b_age = self._num_series(out, b_age_col)
            out["turnover_growth_velocity"] = np.maximum(bt, 0) / (np.maximum(b_age, 0) + 1.0)
            out["business_life_stage"] = np.select([b_age <= 2.0, (b_age > 2.0) & (b_age <= 5.0), (b_age > 5.0) & (b_age <= 10.0), b_age > 10.0], [1, 2, 3, 4], default=0)
            if "owner_age" in out.columns:
                owner_age = self._num_series(out, "owner_age")
                out["owner_experience_velocity"] = b_age / np.maximum(owner_age, 18.0)
                out["founder_inception_age"] = owner_age - b_age

        # 7) Psychological / distress
        opt_df = pd.DataFrame(index=out.index)
        for col in ["attitude_stable_business_environment", "attitude_satisfied_with_achievement", "attitude_more_successful_next_year"]:
            if col in out.columns:
                s = self._norm_str(out[col])
                opt_df[col] = np.where(s.eq("missing"), np.nan, np.where(self._is_current_yes(s), 1.0, 0.0))
        for col in ["attitude_worried_shutdown", "current_problem_cash_flow", "problem_sourcing_money"]:
            if col in out.columns:
                s = self._norm_str(out[col])
                opt_df[col] = np.where(s.eq("missing"), np.nan, np.where(self._is_current_yes(s), -1.0, 0.0))

        out["psychological_optimism_score"] = opt_df.mean(axis=1) if len(opt_df.columns) else np.nan
        out["liquidity_distress_index"] = (-out["psychological_optimism_score"].fillna(0)) * out["burn_rate"].fillna(0).clip(0, 5)

        # 8) Credit / informality interactions
        if "offers_credit_to_customers" in out.columns and "has_loan_account" in out.columns:
            gives_credit = self._is_current_yes(self._norm_str(out["offers_credit_to_customers"]))
            has_formal_loan = self._is_current_access(self._norm_str(out["has_loan_account"]))
            ans_loan = ~self._norm_str(out["has_loan_account"]).eq("missing")
            out["informal_liquidity_trap"] = (gives_credit & (~has_formal_loan) & ans_loan).astype(int)

        if "informal_funding_reliance" in out.columns and "formal_finance_score" in out.columns:
            out["credit_access_vulnerability"] = (1.0 - out["formal_finance_score"].fillna(0)) * (0.5 + out["informal_funding_reliance"].fillna(0))

        return out

    # --- Helpers ---
    def _fit_missingness_reference(self, df):
        if "country" in df.columns and "raw_missing_count" in df.columns:
            self.country_missing_means_ = df.groupby("country")["raw_missing_count"].mean().to_dict()
            self.global_missing_mean_ = float(df["raw_missing_count"].mean())

    def _get_business_age_col(self, df):
        return "business_age" if "business_age" in df.columns else "business_age_years" if "business_age_years" in df.columns else None

    def _num_series(self, df, col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index, dtype=float)

    def _norm_str(self, s): return s.astype("string").str.strip().str.lower().fillna("missing")
    def _is_current_yes(self, s): return self._norm_str(s).isin({"yes", "yes, always", "yes, sometimes"})
    def _is_current_access(self, s): return self._norm_str(s).isin({"yes", "yes, always", "yes, sometimes", "have now"})
    def _is_lost_access(self, s): return self._norm_str(s).eq("used to have but dont have now")

    def _block_current_score(self, df, cols):
        tmp = pd.DataFrame(index=df.index)
        for col in cols:
            if col in df.columns:
                s = self._norm_str(df[col])
                tmp[col] = np.where(s.eq("missing"), np.nan, np.where(self._is_current_access(s), 1.0, 0.0))
        return tmp.mean(axis=1) if tmp.shape[1] > 0 else pd.Series(np.nan, index=df.index, dtype=float)

    def _count_missing(self, df, cols):
        tmp = pd.DataFrame(index=df.index)
        for col in cols:
            if col in df.columns: tmp[col] = self._norm_str(df[col]).eq("missing").astype(int)
        return tmp.sum(axis=1) if tmp.shape[1] > 0 else pd.Series(0, index=df.index, dtype=int)

    def _count_lost_products(self, df, cols):
        tmp = pd.DataFrame(index=df.index)
        for col in cols:
            if col in df.columns: tmp[col] = self._is_lost_access(df[col]).astype(int)
        return tmp.sum(axis=1) if tmp.shape[1] > 0 else pd.Series(0, index=df.index, dtype=int)
    


class FHIAdvancedSignals(BaseEstimator, TransformerMixin):
    """ Dataset-level statistics, K-Means clustering, and MICE Imputation."""

    def __init__(self, random_state=SEED, mice_max_iter=10, use_mice=True, n_clusters=3):
        self.random_state = random_state
        self.mice_max_iter = mice_max_iter
        self.use_mice = use_mice
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        Xc = X.copy()

        # 1. Fit & Create Country Stats / Interactions
        Xc = self._engineer_advanced(Xc, fit_mode=True)

        # 2. Fit K-Means
        self.cluster_cols_ = [c for c in ["personal_income", "business_turnover", "business_expenses"] if c in Xc.columns]
        self.use_clusters_ = len(self.cluster_cols_) == 3
        if self.use_clusters_:
            cluster_df = Xc[self.cluster_cols_].apply(pd.to_numeric, errors="coerce")
            self.cluster_fill_medians_ = cluster_df.median().to_dict()
            X_cluster = np.log1p(cluster_df.fillna(self.cluster_fill_medians_).clip(lower=0))
            self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
            self.kmeans_.fit(X_cluster)
            Xc["financial_cluster"] = self.kmeans_.predict(X_cluster).astype(int)

        # 3. Clean up before MICE
        Xc = self._finalize_output(Xc)
        num_cols = Xc.select_dtypes(include=[np.number]).columns.tolist()

        self.indicator_cols_ = [c for c in num_cols if c.endswith("_is_missing") or c.endswith("_is_zero")]
        do_not_impute = {"relative_missingness", "master_formalization_index", "digital_access_score", "formal_finance_score", "insurance_access_score", "digital_inclusion_score", "psychological_optimism_score", "relative_formalization", "relative_digital_access", "relative_formal_finance", "relative_insurance_access", "relative_inclusion", "personal_income_country_z", "business_expenses_country_z", "business_turnover_country_z", "n_missing_finance_access", "n_missing_insurance", "n_missing_formality", "n_missing_attitudes", "lost_financial_product_count", "business_life_stage", "financial_cluster", "informal_funding_reliance", "informal_funding_loss", "digital_formal_gap", "insurance_need_gap", "insurance_constraint_index", "insurance_gap_x_scale"}

        self.mice_cols_ = [c for c in num_cols if c not in self.indicator_cols_ and c not in do_not_impute and Xc[c].notna().any()]
        self.fallback_medians_ = Xc[self.mice_cols_].median(numeric_only=True).to_dict() if self.mice_cols_ else {}

        # 4. Fit MICE
        if self.use_mice and self.mice_cols_:
            self.imputer_ = IterativeImputer(max_iter=self.mice_max_iter, random_state=self.random_state)
            self.imputer_.fit(Xc[self.mice_cols_])

        return self

    def transform(self, X):
        Xt = X.copy()

        # 1. Apply Country Stats / Interactions
        Xt = self._engineer_advanced(Xt, fit_mode=False)

        # 2. Apply K-Means
        if getattr(self, "use_clusters_", False):
            cluster_df = Xt[self.cluster_cols_].apply(pd.to_numeric, errors="coerce")
            X_cluster = np.log1p(cluster_df.fillna(self.cluster_fill_medians_).clip(lower=0))
            Xt["financial_cluster"] = self.kmeans_.predict(X_cluster).astype(int)

        # 3. Clean up before MICE
        Xt = self._finalize_output(Xt)

        # 4. Apply MICE
        if self.use_mice and hasattr(self, "imputer_") and self.mice_cols_:
            for c in self.mice_cols_:
                if c not in Xt.columns: Xt[c] = np.nan
            Xt[self.mice_cols_] = self.imputer_.transform(Xt[self.mice_cols_])
        elif self.mice_cols_:
            for c in self.mice_cols_:
                if c not in Xt.columns: Xt[c] = np.nan
            Xt[self.mice_cols_] = Xt[self.mice_cols_].fillna(self.fallback_medians_)

        for c in getattr(self, "indicator_cols_", []):
            if c in Xt.columns: Xt[c] = Xt[c].fillna(0).astype(int)

        return Xt

    def _engineer_advanced(self, out, fit_mode=False):
        b_age_col = "business_age" if "business_age" in out.columns else "business_age_years" if "business_age_years" in out.columns else None

        if fit_mode:
            self._fit_country_reference_stats(out)

        # Country Z-Scores and Relatives
        if "country" in out.columns and hasattr(self, "country_feature_stats_"):
            out["relative_formalization"] = self._relative_to_country_mean(out, "master_formalization_index")
            out["relative_digital_access"] = self._relative_to_country_mean(out, "digital_access_score")
            out["relative_formal_finance"] = self._relative_to_country_mean(out, "formal_finance_score")
            out["relative_insurance_access"] = self._relative_to_country_mean(out, "insurance_access_score")
            out["relative_inclusion"] = self._relative_to_country_mean(out, "digital_inclusion_score")
            out["personal_income_country_z"] = self._country_zscore(out, "personal_income_log")
            out["business_expenses_country_z"] = self._country_zscore(out, "business_expenses_log")
            out["business_turnover_country_z"] = self._country_zscore(out, "business_turnover_log")

        # High-Value Interactions
        out["digital_x_formality"] = out["digital_access_score"].fillna(0) * out["master_formalization_index"].fillna(0)
        out["finance_x_formality"] = out["formal_finance_score"].fillna(0) * out["master_formalization_index"].fillna(0)
        out["margin_x_credit_constraint"] = (1.0 - out["burn_rate"].fillna(0).clip(upper=1.0)) * (1.0 - out["formal_finance_score"].fillna(0))

        out["maturity_x_inclusion_gap"] = np.log1p(pd.to_numeric(out[b_age_col], errors="coerce").fillna(0).clip(lower=0)) * (1.0 - out["digital_inclusion_score"].fillna(0)) if b_age_col is not None else np.nan
        out["scale_x_finance_missing"] = out["business_turnover_country_z"].fillna(0) * out["n_missing_finance_access"].fillna(0) if "business_turnover_country_z" in out.columns else np.nan
        out["insurance_x_scale"] = out["insurance_access_score"].fillna(0) * out["business_turnover_country_z"].fillna(0) if "business_turnover_country_z" in out.columns else np.nan
        out["insurance_gap_x_scale"] = out["insurance_need_gap"].fillna(0) * out["business_turnover_country_z"].fillna(0) if "insurance_need_gap" in out.columns and "business_turnover_country_z" in out.columns else np.nan

        # Rescue Flags
        if "keeps_financial_records" in out.columns and "business_turnover_country_z" in out.columns:
            no_rec = self._norm_str(out["keeps_financial_records"]).isin(["no", "missing"])
            top_earner = out["business_turnover_country_z"].fillna(0) > 0.70
            out["scale_infrastructure_mismatch"] = (top_earner & no_rec).astype(int)

        if "business_turnover" in out.columns and "raw_missing_count" in out.columns:
            bt = pd.to_numeric(out["business_turnover"], errors="coerce").fillna(0)
            miss = pd.to_numeric(out["raw_missing_count"], errors="coerce").fillna(0)
            out["severe_informal_micro"] = ((bt < 3000) & (miss >= 10)).astype(int)

        if 'business_turnover' in out.columns and 'keeps_financial_records' in out.columns:
            is_wealthy = pd.to_numeric(out['business_turnover'], errors="coerce").fillna(0) > 10000
            no_records = out['keeps_financial_records'] == 'missing'
            out['wealth_infrastructure_contradiction'] = (is_wealthy & no_records).astype(int)

        if "business_turnover_country_z" in out.columns and "has_internet_banking" in out.columns and b_age_col is not None:
            has_net = self._is_current_access(self._norm_str(out["has_internet_banking"]))
            no_ins = pd.Series(True, index=out.index)
            if "has_insurance" in out.columns:
                no_ins = ~self._is_current_access(self._norm_str(out["has_insurance"]))
            legacy = pd.to_numeric(out[b_age_col], errors="coerce").fillna(0) > 5.0
            top_15 = out["business_turnover_country_z"].fillna(0) > 1.2
            out["legacy_titan_unbanked"] = (top_15 & legacy & (~has_net) & no_ins).astype(int)

        if "keeps_financial_records" in out.columns and "has_internet_banking" in out.columns and "compliance_income_tax" in out.columns:
            rec_miss = self._norm_str(out["keeps_financial_records"]).eq("missing")
            has_bank = self._is_current_access(self._norm_str(out["has_internet_banking"]))
            pays_tax = self._is_current_yes(self._norm_str(out["compliance_income_tax"]))
            out["rescued_formal_business"] = (rec_miss & (has_bank | pays_tax)).astype(int)

        return out

    def _finalize_output(self, df):
        out = df.copy()
        drop_cols = ["Target", "stratify_col", "ID", "business_turnover", "personal_income", "business_expenses"]
        return out.drop(columns=drop_cols, errors="ignore")

    # --- Helpers ---
    def _fit_country_reference_stats(self, df):
        self.country_feature_stats_ = {}
        if "country" not in df.columns: return
        stat_cols = ["master_formalization_index", "digital_access_score", "formal_finance_score", "insurance_access_score", "digital_inclusion_score", "personal_income_log", "business_expenses_log", "business_turnover_log"]
        for col in stat_cols:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                tmp = pd.DataFrame({"country": df["country"], "_v": s})
                global_std = float(s.std(ddof=0)) if s.notna().sum() > 1 else 1.0
                self.country_feature_stats_[col] = {
                    "mean_map": tmp.groupby("country")["_v"].mean().to_dict(),
                    "std_map": tmp.groupby("country")["_v"].std(ddof=0).to_dict(),
                    "global_mean": float(s.mean()) if s.notna().any() else 0.0,
                    "global_std": global_std if np.isfinite(global_std) and global_std > 0 else 1.0,
                }

    def _relative_to_country_mean(self, df, col):
        if col not in df.columns or "country" not in df.columns or col not in getattr(self, 'country_feature_stats_', {}): return np.nan
        stats = self.country_feature_stats_[col]
        c_mean = df["country"].map(stats["mean_map"]).fillna(stats["global_mean"])
        return pd.to_numeric(df[col], errors="coerce") - c_mean

    def _country_zscore(self, df, col):
        if col not in df.columns or "country" not in df.columns or col not in getattr(self, 'country_feature_stats_', {}): return np.nan
        stats = self.country_feature_stats_[col]
        s = pd.to_numeric(df[col], errors="coerce")
        c_mean = df["country"].map(stats["mean_map"]).fillna(stats["global_mean"])
        c_std = df["country"].map(stats["std_map"]).replace(0, np.nan).fillna(stats["global_std"]).replace(0, 1.0)
        return (s - c_mean) / c_std

    def _norm_str(self, s): return s.astype("string").str.strip().str.lower().fillna("missing")
    def _is_current_yes(self, s): return self._norm_str(s).isin({"yes", "yes, always", "yes, sometimes"})
    def _is_current_access(self, s): return self._norm_str(s).isin({"yes", "yes, always", "yes, sometimes", "have now"})

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold=0.98, variance_threshold=0.0):
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.selector = VarianceThreshold(threshold=variance_threshold)
        self.drop_corr_cols_ = []

    def fit(self, X, y=None):
        # 1. Fit Variance Threshold (Identifies constant columns)
        # We only check numeric columns
        X_num = X.select_dtypes(include=[np.number])
        self.selector.fit(X_num)

        # Get columns that passed the variance check
        self.valid_var_cols_ = X_num.columns[self.selector.get_support()].tolist()

        # 2. Identify Correlated Columns (Multicollinearity)
        X_clean = X_num[self.valid_var_cols_]

        # Calculate correlation matrix
        corr_matrix = X_clean.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than threshold
        self.drop_corr_cols_ = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]

        if self.drop_corr_cols_:
            print(f"Dropping {len(self.drop_corr_cols_)} highly correlated features.")

        return self

    def transform(self, X):
        X_new = X.copy()

        # 1. Drop Zero Variance Columns (if they exist in X)
        cols_to_keep = [c for c in X_new.columns if c in self.valid_var_cols_ or c not in X.select_dtypes(include=[np.number]).columns]
        X_new = X_new[cols_to_keep]

        # 2. Drop Correlated Columns
        final_drop = [c for c in self.drop_corr_cols_ if c in X_new.columns]
        print("dropped columns : ",final_drop)
        X_new = X_new.drop(columns=final_drop)

        return X_new
    

class FeatureNameSanitizer(BaseEstimator, TransformerMixin):
    """
    This transformer replaces all non-alphanumeric characters with underscores.

    Additionally, if the sanitization causes two distinct columns to end up with the
    exact same name (a collision), it safely drops the redundant duplicates, keeping
    only the first instance.
    """

    def fit(self, X, y=None):
        """
        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable. Defaults to None.

        Returns:
            self: Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Sanitizes column names and removes resulting duplicates.

        Args:
            X (pd.DataFrame): The input DataFrame whose columns need sanitization.

        Returns:
            pd.DataFrame: A new DataFrame with clean, unique column names.
        """
        # Replace non-alphanumeric chars with underscore
        clean_cols = [re.sub(r'[^\w]', '_', str(col)) for col in X.columns]

        # Identify Unique Columns vs Duplicates
        seen = {}
        keep_indices = []
        final_names = []

        for i, col in enumerate(clean_cols):
            if col not in seen:
                seen[col] = True
                keep_indices.append(i)
                final_names.append(col)
            else:
                # If we've seen this name before, it means we have a collision.
                # Since the data likely represents the same concept (just different punctuation),
                # we drop this redundant column.
                pass

        # Filter the DataFrame to keep only the first unique instances
        X_clean = X.iloc[:, keep_indices].copy()
        X_clean.columns = final_names

        return X_clean
    

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols  # IMPORTANT: no list(), no copy(), no sorting

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cols = [] if self.cols is None else list(self.cols)  # convert HERE
        return X.drop(columns=cols, errors="ignore")
    

def custom_smote_ratios(y):
    """
    Dynamically maps SMOTE ratios based on class frequencies.
    """
    target_stats = Counter(y)

    # Sort the classes by how many rows they have (Highest count to lowest count)
    sorted_classes = sorted(target_stats.items(), key=lambda item: item[1], reverse=True)
    #returns [(np.int64(1), 6280 (low)), (np.int64(2), 2868(medium)), (np.int64(0), 470(high))]

    # Extract the dynamic labels based on their rank
    low_label = sorted_classes[0][0]     # The most frequent class
    medium_label = sorted_classes[1][0]  # The second most frequent class
    high_label = sorted_classes[2][0]    # The least frequent class

    majority_count = sorted_classes[0][1]

    # Apply our 100% / 60% / 25% rule using the dynamically found labels
    return {
        low_label: target_stats[low_label],           # Keep original
        medium_label: int(majority_count * 0.60),     # Pad to 60% of Majority
        high_label: int(majority_count * 0.25)        # Pad to 25% of Majority
    }