```python
import numpy as np
from numpy import special as nps
from sklearn.isotonic import IsotonicRegression

class CDFPITCalibrator:
    """
    Post-hoc CDF calibration via Probability Integral Transform (PIT) +
    sklearn's IsotonicRegression.

    Given per-sample predicted quantiles at fixed levels `alphas`, we:
      1) compute PIT values U = F_hat(y|x) on a calibration set;
      2) fit a monotone map c: [0,1] -> [0,1] by isotonic regression so that c(U) ~ Uniform(0,1);
      3) define calibrated CDF as F_tilde(y|x) = c( F_hat(y|x) ), and obtain calibrated quantiles by inversion.

    Notes:
      • We lightly enforce non-crossing quantiles by cumulative max in score space.
      • We ignore zero-inflation here (unconditional CDF only), per your request.
    """
    def __init__(self, alphas):
        self.alphas = np.asarray(alphas, float)
        self.iso = None            # isotonic c(U)
        self._x_thr = None         # X_thresholds_ (domain knots of c)
        self._y_thr = None         # y_thresholds_ (range knots of c)

    def _mk_monotone(self, q_row):
        """Enforce non-decreasing quantiles across `alphas` for one sample."""
        return np.maximum.accumulate(np.asarray(q_row, float))

    def _pit(self, y, q_row):
        """Compute F_hat(y) from a single row of quantiles by linear interpolation."""
        q_row = self._mk_monotone(q_row)
        return np.interp(y, q_row, self.alphas, left=0.0, right=1.0)

    def fit(self, Q_cal, y_cal):
        """
        Fit isotonic c on calibration set.
          Q_cal: (n_cal, n_levels) predicted (possibly crossing) quantiles
          y_cal: (n_cal,) true targets
        """
        Q_cal = np.asarray(Q_cal, float)
        y_cal = np.asarray(y_cal, float).ravel()

        # PIT values on calibration set
        U = np.array([self._pit(y_cal[i], Q_cal[i]) for i in range(len(y_cal))])

        # Empirical CDF targets Z in (0,1): z_i = (rank(u_i)-0.5)/n
        ranks = U.argsort(kind="mergesort").argsort()
        Z = (ranks + 0.5) / len(U)

        # Fit monotone map c: U -> Z
        self.iso = IsotonicRegression(increasing=True, out_of_bounds="clip",
                                      y_min=0.0, y_max=1.0).fit(U, Z)
        self._x_thr = np.asarray(self.iso.X_thresholds_, float)
        self._y_thr = np.asarray(self.iso.y_thresholds_, float)
        return self

    def calibrate_cdf(self, q_row, y):
        """Return calibrated CDF F_tilde(y) for one sample."""
        F_unc = self._pit(y, q_row)
        return self.iso.predict(np.asarray(F_unc, float))

    def calibrated_quantiles(self, q_row, target_alphas):
        """Return calibrated quantiles by inverting c and the (monotone) quantile function."""
        q_row = self._mk_monotone(q_row)
        target_alphas = np.asarray(target_alphas, float)
        # u* = c^{-1}(alpha) via swapping isotonic axes (piecewise-linear inverse)
        u_star = np.interp(target_alphas, self._y_thr, self._x_thr,
                           left=self._x_thr[0], right=self._x_thr[-1])
        return np.interp(u_star, self.alphas, q_row,
                         left=q_row[0], right=q_row[-1])


if __name__ == "__main__":
    # --- Minimal example with synthetic data ---
    rng = np.random.default_rng(7)

    # Quantile grid your model was trained for
    alphas = np.linspace(0.01, 0.99, 99)

    # Calibration data (X not used by calibrator; here to generate y)
    n_cal = 2000
    X = rng.uniform(-1, 1, size=n_cal)
    mu = 2.0 + 3.0 * X
    sigma = 0.5 + 0.5 * np.abs(X)

    # True standard normal quantile function via erfinv (no SciPy dependency)
    z = np.sqrt(2.0) * nps.erfinv(2.0 * alphas - 1.0)  # shape (n_levels,)

    # Simulate a miscalibrated model's quantile outputs by warping alpha -> alpha_hat
    alpha_hat = 0.5 + 0.9 * (alphas - 0.5) + 0.05 * np.sin(2 * np.pi * alphas)
    alpha_hat = np.clip(alpha_hat, 1e-6, 1 - 1e-6)
    z_hat = np.sqrt(2.0) * nps.erfinv(2.0 * alpha_hat - 1.0)

    # Per-sample predicted quantiles (possibly miscalibrated)
    Q_cal = mu[:, None] + sigma[:, None] * z_hat[None, :]   # (n_cal, n_levels)

    # Ground-truth outcomes for calibration
    y_cal = mu + sigma * rng.standard_normal(n_cal)

    # Fit PIT calibrator
    cal = CDFPITCalibrator(alphas).fit(Q_cal, y_cal)

    # Pick one sample and see calibrated vs uncalibrated quantiles at 10%, 50%, 90%
    i = 0
    target = np.array([0.1, 0.5, 0.9])
    q_unc = np.interp(target, alphas, np.maximum.accumulate(Q_cal[i]))
    q_cal = cal.calibrated_quantiles(Q_cal[i], target)

    print("Uncalibrated quantiles @ [0.1, 0.5, 0.9]:", np.round(q_unc, 3))
    print("Calibrated   quantiles @ [0.1, 0.5, 0.9]:", np.round(q_cal, 3))

    # Quick PIT sanity check: U should be ~Uniform(0,1) after calibration
    U_unc = np.array([np.interp(y_cal[j], np.maximum.accumulate(Q_cal[j]), alphas, left=0.0, right=1.0)
                      for j in range(n_cal)])
    U_cal = cal.iso.predict(U_unc)
    print("PIT mean (uncal, cal):", float(U_unc.mean()), float(U_cal.mean()))
    print("PIT std  (uncal, cal):", float(U_unc.std()),  float(U_cal.std()))
```
