import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import random
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)

class UnsupervisedPatternExtractor(BaseEstimator, TransformerMixin):
    """
    A custom Scikit-Learn transformer that extracts unsupervised macroscopic 
    patterns from numeric features to create new meta-features for downstream models.
    
    This transformer uses Principal Component Analysis (PCA) for dimensionality 
    reduction and Gaussian Mixture Models (GMM) for soft clustering. It safely 
    handles zero-variance columns and includes fallback mechanisms to prevent 
    crashing on sparse or highly collinear folds during cross-validation.

    Parameters
    ----------
    n_clusters : int, default=3
        The target number of underlying patterns/clusters to search for using GMM.
    use_gmm : bool, default=True
        Whether to fit a Gaussian Mixture Model to extract cluster probabilities 
        and outlier scores.
    use_pca : bool, default=True
        Whether to apply PCA (retaining 97% of variance) prior to GMM clustering 
        to reduce noise and prevent the curse of dimensionality.

    Attributes
    ----------
    valid_cols_ : list of str
        The numeric columns retained after dropping constant (zero-variance) features.
    pca_model : sklearn.decomposition.PCA or None
        The fitted PCA model (if use_pca=True).
    gmm_model : sklearn.mixture.GaussianMixture or None
        The fitted GMM model (if use_gmm=True).
    """
    def __init__(self, n_clusters=3, use_gmm=True, use_pca=True):
        self.n_clusters = n_clusters
        self.use_gmm = use_gmm
        self.use_pca = use_pca
        self.gmm_model = None
        self.pca_model = None

    def fit(self, X, y=None):
        # 1. Select numeric features & Fill NA
        X_num = X.select_dtypes(include=[np.number]).fillna(0)

        # Safety: Drop columns with 0 variance (constants)
        # These cause GMM to crash immediately
        self.valid_cols_ = X_num.columns[X_num.var() > 1e-6].tolist()
        X_clean = X_num[self.valid_cols_]

        # 2. PCA
        if self.use_pca:
            self.pca_model = PCA(n_components=0.97, random_state=SEED)
            X_reduced = self.pca_model.fit_transform(X_clean)
        else:
            X_reduced = X_clean

        # 3. GMM with SAFETY FEATURES
        if self.use_gmm:
            try:
                # Attempt 1: Standard Fit with Regularization
                self.gmm_model = GaussianMixture(
                    n_components=self.n_clusters,
                    random_state=SEED,
                    n_init=3,
                    reg_covar=1e-4  # <--- CRITICAL FIX: Adds stability
                )
                self.gmm_model.fit(X_reduced)

            except Exception:
                # Attempt 2: Fallback (Reduce clusters if data is too sparse)
                print(f"⚠️ GMM failed with {self.n_clusters} clusters. Retrying with 2...")
                self.gmm_model = GaussianMixture(
                    n_components=2,
                    random_state=SEED,
                    n_init=2,
                    reg_covar=1e-3  # Stronger regularization
                )
                self.gmm_model.fit(X_reduced)

        return self

    def transform(self, X):
        X_new = X.copy()
        X_num = X.select_dtypes(include=[np.number]).fillna(0)

        # Handle case where columns are missing in test that were present in train
        valid_cols = [c for c in self.valid_cols_ if c in X_num.columns]
        X_clean = X_num[valid_cols]

        # If dimensions mismatch (e.g. dropped cols), PCA will fail.
        # We fill missing valid cols with 0 to match fit shape.
        for c in self.valid_cols_:
            if c not in X_clean.columns:
                X_clean[c] = 0
        X_clean = X_clean[self.valid_cols_]

        if self.use_pca:
            X_reduced = self.pca_model.transform(X_clean)
        else:
            X_reduced = X_clean

        if self.use_gmm and hasattr(self, 'gmm_model'):
            probs = self.gmm_model.predict_proba(X_reduced)

            # Dynamic loop based on actual fitted components (in case of fallback)
            actual_clusters = self.gmm_model.n_components
            for i in range(actual_clusters):
                X_new[f"pattern_prob_{i}"] = probs[:, i]

            X_new["pattern_outlier_score"] = self.gmm_model.score_samples(X_reduced)
            X_new["pattern_id"] = probs.argmax(axis=1)

        return X_new