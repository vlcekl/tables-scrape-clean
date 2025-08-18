```python
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

class CDFPITCalibrator:
    """
    Post-hoc CDF calibration via PIT + sklearn IsotonicRegression.
    Given per-sample predicted quantiles at fixed levels `alphas`, we fit a
    monotone map c:[0,1]->[0,1] so that c(F_hat(y|x)) ~ Uniform(0,1), then
    define calibrated CDF as c(F_hat). Calibrated quantiles come from inversion.
    """
    def __init__(self, alphas):
        self.alphas = np.asarray(alphas, float)
        self.iso = None
        self._x_thr = None
        self._y_thr = None

    def _mk_monotone(self, q_row):
        # L2-optimal monotone projection along the quantile axis using IsotonicRegression
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(self.alphas, np.asarray(q_row, float))
        return iso.predict(self.alphas)

    def _pit(self, y, q_row):
        # Compute F_hat(y) by linear interpolation through (quantile, alpha) pairs.
        q_row = self._mk_monotone(q_row)
        return np.interp(y, q_row, self.alphas, left=0.0, right=1.0)

    def fit(self, Q_cal, y_cal):
        Q_cal = np.asarray(Q_cal, float)
        y_cal = np.asarray(y_cal, float).ravel()
        # PIT values U_i = F_hat(y_i)
        U = np.array([self._pit(y_cal[i], Q_cal[i]) for i in range(len(y_cal))])
        # Empirical CDF targets Z in (0,1): z_i = (rank(u_i)-0.5)/n
        ranks = U.argsort(kind="mergesort").argsort()
        Z = (ranks + 0.5) / len(U)
        # Fit isotonic c: U -> Z
        self.iso = IsotonicRegression(increasing=True, out_of_bounds="clip",
                                      y_min=0.0, y_max=1.0).fit(U, Z)
        self._x_thr = np.asarray(self.iso.X_thresholds_, float)
        self._y_thr = np.asarray(self.iso.y_thresholds_, float)
        return self

    def calibrate_cdf(self, q_row, y):
        F_unc = self._pit(y, q_row)
        return self.iso.predict(np.asarray(F_unc, float))

    def calibrated_quantiles(self, q_row, target_alphas):
        q_row = self._mk_monotone(q_row)
        target_alphas = np.asarray(target_alphas, float)
        # c^{-1}(alpha) via swapping isotonic axes
        u_star = np.interp(target_alphas, self._y_thr, self._x_thr,
                           left=self._x_thr[0], right=self._x_thr[-1])
        return np.interp(u_star, self.alphas, q_row,
                         left=q_row[0], right=q_row[-1])


def _fit_quantile_gbm_models(X_train, y_train, alphas, **gbm_kwargs):
    models = []
    for a in alphas:
        m = GradientBoostingRegressor(loss='quantile', alpha=float(a), **gbm_kwargs)
        m.fit(X_train, y_train)
        models.append(m)
    return models


def _predict_quantiles(models, X, alphas):
    # Stack predictions column-wise: shape (n_samples, n_levels)
    Q = np.column_stack([m.predict(X) for m in models])
    # Project each row onto the monotone cone along alpha-axis via L2 isotonic
    A = np.asarray(alphas, float)
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    out = np.empty_like(Q)
    for i in range(Q.shape[0]):
        iso.fit(A, Q[i])
        out[i] = iso.predict(A)
    return out


def _quantile_ece(y, Q, alphas):
    # Expected calibration error for quantiles: mean_j | P(Y<=q_j) - alpha_j |
    cover = [(y <= Q[:, j]).mean() for j in range(len(alphas))]
    return float(np.mean(np.abs(np.asarray(cover) - np.asarray(alphas))))



def summarize_cdf_ensemble(Q_ens, alphas, y_grid=None, n_grid=201, band_probs=(0.2, 0.5, 0.8)):
    """
    Summarize an ensemble of calibrated CDFs provided as a dict of per-model
    quantile arrays. Returns the mean CDF and ensemble quantile bands across models.

    Parameters
    ----------
    Q_ens : dict[str, ndarray], each value shape (n_samples, n_alphas)
        For each model id -> matrix of per-sample quantiles at `alphas`.
        Assumes inputs are already non-crossing along the alpha axis.
    alphas : array-like, shape (n_alphas,)
        Increasing quantile levels used to construct the CDFs.
    y_grid : array-like, optional
        Grid of y-values at which to evaluate/aggregate CDFs. If None, uses a
        linear grid spanning the global min..max of all quantiles with `n_grid` points.
    n_grid : int
        Number of grid points when y_grid is None.
    band_probs : tuple of floats in (0,1)
        Ensemble quantiles to report across models at each y (e.g., (0.2, 0.5, 0.8)).

    Returns
    -------
    y_eval : ndarray, shape (n_y,)
        The y-grid used for evaluation.
    mean_cdf : ndarray, shape (n_samples, n_y)
        Mean CDF across models at each y.
    band_cdfs : dict[float -> ndarray], each of shape (n_samples, n_y)
        For each probability p in band_probs, band_cdfs[p] holds the p-quantile
        CDF across models at each y.
    """
    A = np.asarray(alphas, float)
    mats = list(Q_ens.values())  # list of (n_samples, n_alphas)
    M = len(mats)
    N, K = mats[0].shape

    # Evaluation grid
    if y_grid is None:
        all_q_min = min(float(m.min()) for m in mats)
        all_q_max = max(float(m.max()) for m in mats)
        y_eval = np.linspace(all_q_min, all_q_max, int(n_grid))
    else:
        y_eval = np.asarray(y_grid, float)

    n_y = y_eval.size

    # Compute CDFs per model/sample via interpolation of (quantile, alpha) pairs
    F_stack = np.empty((M, N, n_y), float)
    for m_idx, Q in enumerate(mats):
        for i in range(N):
            F_stack[m_idx, i] = np.interp(y_eval, Q[i], A, left=0.0, right=1.0)

    # Aggregate across models
    mean_cdf = F_stack.mean(axis=0)
    band_cdfs = {p: np.quantile(F_stack, p, axis=0, method='linear') for p in band_probs}

    return y_eval, mean_cdf, band_cdfs


if __name__ == "__main__":
    # ==== 1) Generate synthetic data and split into train / cal / test ====
    rng = np.random.default_rng(42)
    n, d = 9000, 5
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    # Nonlinear mean and heteroscedastic noise
    mu = 1.5 + 1.2*X[:,0] - 2.0*X[:,1] + 0.5*np.sin(np.pi*X[:,2])
    sigma = 0.4 + 0.6/(1+np.exp(-2*X[:,3])) + 0.3*np.abs(X[:,4])
    y = mu + sigma * rng.standard_normal(n)

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.40, random_state=0)
    X_cal, X_te,  y_cal, y_te  = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=1)

    # ==== 2) Fit a quantile regressor family (GBM per alpha) ====
    alphas = np.linspace(0.05, 0.95, 19)
    gbm_kwargs = dict(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)
    models = _fit_quantile_gbm_models(X_tr, y_tr, alphas, **gbm_kwargs)

    # Predicted quantiles on calibration and test sets
    Q_cal = _predict_quantiles(models, X_cal, alphas)
    Q_te  = _predict_quantiles(models, X_te, alphas)

    # ==== 3) Fit PIT calibrator on calibration set ====
    cal = CDFPITCalibrator(alphas).fit(Q_cal, y_cal)

    # ==== 4) Evaluate on test set ====
    # 4a) Quantile ECE before/after calibration
    # Uncalibrated quantiles already non-crossing via _predict_quantiles
    ece_unc = _quantile_ece(y_te, Q_te, alphas)
    # Calibrated quantiles row-wise
    Q_te_cal = np.row_stack([cal.calibrated_quantiles(Q_te[i], alphas) for i in range(len(y_te))])
    ece_cal = _quantile_ece(y_te, Q_te_cal, alphas)

    # 4b) PIT stats before/after calibration
    U_unc = np.array([np.interp(y_te[i], Q_te[i], alphas, left=0.0, right=1.0) for i in range(len(y_te))])
    U_cal = cal.iso.predict(U_unc)

    print("Quantile ECE (uncal / cal): {:.4f} / {:.4f}".format(ece_unc, ece_cal))
    print("PIT mean   (uncal / cal): {:.3f} / {:.3f}".format(U_unc.mean(), U_cal.mean()))
    print("PIT std    (uncal / cal): {:.3f} / {:.3f}".format(U_unc.std(),  U_cal.std()))

    # Optional: show a few quantile comparisons for first 3 test points
    idx = [0, 1, 2]
    show = np.array([0.1, 0.5, 0.9])
    for i in idx:
        q_unc = np.interp(show, alphas, Q_te[i])
        q_cal = cal.calibrated_quantiles(Q_te[i], show)
        print(f"Sample {i}: uncal {np.round(q_unc,3)} | cal {np.round(q_cal,3)}")
```
