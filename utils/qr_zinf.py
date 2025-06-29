#%%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier, LGBMRegressor

class ZeroInflatedConformal(BaseEstimator, RegressorMixin):
    """
    Zero-inflated conformal predictor for non-negative targets.
    Produces mixture quantiles over a sequence of alpha levels,
    with optional isotonic regression to enforce non-decreasing CDF.
    """
    def __init__(self, quantile_alphas=None, calibrate_size=0.2,
                 random_state=42, enforce_monotonic=True, **lgbm_kwargs):
        # sequence of quantile levels (e.g., np.linspace(0.01,0.99,99))
        self.quantile_alphas = quantile_alphas if quantile_alphas is not None else np.linspace(0.01, 0.99, 99)
        self.calibrate_size = calibrate_size
        self.random_state = random_state
        self.enforce_monotonic = enforce_monotonic
        self.lgbm_kwargs = lgbm_kwargs

    def fit(self, X, y):
        # Split into training and calibration
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y, test_size=self.calibrate_size,
            random_state=self.random_state)

        # 1) Fit zero-mass classifier
        z_train = (y_train == 0).astype(int)
        self.zero_model_ = LGBMClassifier(objective='binary', **self.lgbm_kwargs)
        self.zero_model_.fit(X_train, z_train)
        # Calibration zero probs and scores
        pi_calib = self.zero_model_.predict_proba(X_calib)[:, 1]
        z_true_calib = (y_calib == 0).astype(int)
        self.z_thresholds_ = {
            alpha: np.quantile(np.abs(z_true_calib - pi_calib), 1 - alpha)
            for alpha in self.quantile_alphas
        }

        # 2) Fit quantile regressors on positives
        pos_idx = y_train > 0
        X_pos, y_pos = X_train[pos_idx], y_train[pos_idx]
        self.lower_models_, self.upper_models_ = {}, {}
        for alpha in self.quantile_alphas:
            lm = LGBMRegressor(objective='quantile', alpha=alpha,
                                **self.lgbm_kwargs)
            um = LGBMRegressor(objective='quantile', alpha=1-alpha,
                                **self.lgbm_kwargs)
            lm.fit(X_pos, y_pos)
            um.fit(X_pos, y_pos)
            self.lower_models_[alpha] = lm
            self.upper_models_[alpha] = um

        # Calibration conformity thresholds for positives
        pos_calib_idx = y_calib > 0
        X_pos_calib, y_pos_calib = X_calib[pos_calib_idx], y_calib[pos_calib_idx]
        self.quantile_thresholds_ = {}
        for alpha in self.quantile_alphas:
            ql = self.lower_models_[alpha].predict(X_pos_calib)
            qu = self.upper_models_[alpha].predict(X_pos_calib)
            residuals = np.maximum(ql - y_pos_calib, y_pos_calib - qu)
            self.quantile_thresholds_[alpha] = np.quantile(residuals, 1 - alpha)

        return self

    def predict(self, X_new):
        """
        Return mixture quantiles: array of shape (n_alphas, n_samples),
        where each entry is the mixture quantile at alpha level.
        Applies isotonic regression across quantile levels if enabled.
        """
        pi = self.zero_model_.predict_proba(X_new)[:, 1]
        n_samples = X_new.shape[0]
        n_alphas = len(self.quantile_alphas)
        result = np.zeros((n_alphas, n_samples))

        # Build raw mixture quantiles
        for i, alpha in enumerate(self.quantile_alphas):
            z_thr = self.z_thresholds_[alpha]
            pi_adj = np.clip(pi, z_thr, 1 - z_thr)

            ql = self.lower_models_[alpha].predict(X_new)
            qu = self.upper_models_[alpha].predict(X_new)
            c = self.quantile_thresholds_[alpha]

            lower = np.maximum(ql - c, 0)
            upper = qu + c
            # midpoint of positive conformal interval
            pos_mid = (lower + upper) / 2
            # mixture quantile: zero for alpha <= pi_adj
            result[i] = np.where(alpha <= pi_adj, 0, pos_mid)

        # Enforce non-decreasing quantiles with isotonic regression per sample
        if self.enforce_monotonic:
            ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
            for j in range(n_samples):
                result[:, j] = ir.fit_transform(self.quantile_alphas, result[:, j])

        return result


#%%
if __name__ == "__main__":
    # Example: synthetic data demonstration
    from sklearn.datasets import make_regression
    # generate data with zeros
    X, y_cont = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)
    # inject zero-inflation: about 30% zeros
    mask = np.random.RandomState(42).rand(1000) < 0.3
    y = y_cont.copy()
    y[mask] = 0

    # Split off a test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Initialize and fit the model
    model = ZeroInflatedConformal(
        quantile_alphas=np.linspace(0.01, 0.99, 99),
        calibrate_size=0.2,
        enforce_monotonic=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predict mixture quantiles on test set
    mixture_q = model.predict(X_test)

    # Display a few quantiles for the first test sample
    print("Quantile levels:", model.quantile_alphas)
    print("Mixture quantiles (first sample):", mixture_q[:, 0])
    print("Mixture quantiles (second sample):", mixture_q[:, 1])
    print("Mixture quantiles (second sample):", mixture_q[:, 10])

# %%
