import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.lag import Lag
from sktime.forecasting.compose import ForecastingPipeline, make_reduction
from sktime.forecasting.model_selection import CutoffSplitter, ForecastingGridSearchCV
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

# --- Two-stage feature selectors ---
class AllRelevantSelector(BaseEstimator, TransformerMixin):
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def fit(self, X, y=None):
        gs = GridSearchCV(LGBMRegressor(), self.param_grid, cv=3, n_jobs=-1)
        gs.fit(X, y)
        self.best_params_ = gs.best_params_
        selector = SelectFromModel(gs.best_estimator_, threshold='median')
        selector.fit(X, y)
        self.support_ = selector.get_support()
        return self

    def transform(self, X):
        return X[:, self.support_]

class RedundancyEliminator(BaseEstimator, TransformerMixin):
    def __init__(self, corr_threshold=0.9, param_grid=None):
        self.corr_threshold = corr_threshold
        self.param_grid = param_grid or {}

    def fit(self, X, y=None):
        if self.param_grid:
            gs = GridSearchCV(LGBMRegressor(), self.param_grid, cv=3, n_jobs=-1)
            gs.fit(X, y)
            self.best_params_ = gs.best_params_
        else:
            self.best_params_ = {}
        corr = np.abs(np.corrcoef(X, rowvar=False))
        to_remove = {j for i in range(corr.shape[0]) for j in range(i) if corr[i, j] > self.corr_threshold}
        self.support_ = np.array([i not in to_remove for i in range(X.shape[1])])
        return self

    def transform(self, X):
        return X[:, self.support_]

# --- FoldAggregator using CutoffSplitter ---
class FoldAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline, cv, n_jobs=1):
        self.pipeline = pipeline
        self.cv = cv
        self.n_jobs = n_jobs

    def _fit_one(self, X, y, train_idx, test_idx, fold_id):
        pipe = clone(self.pipeline)
        X_tr, y_tr = X[train_idx], y[train_idx]
        pipe.fit(X_tr, y_tr)
        return pipe.named_steps['redund'].support_

    def fit(self, X, y=None):
        splits = list(self.cv.split(y))
        masks = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one)(X, y, tr, ts, i)
            for i, (tr, ts) in enumerate(splits)
        )
        self.masks_ = np.vstack(masks)
        self.consensus_mask_ = self.masks_.all(axis=0)
        return self

    def transform(self, X):
        return X[:, self.consensus_mask_]

# --- Main script example ---
if __name__ == '__main__':
    # Assume y_panel and X_panel are loaded pandas objects with MultiIndex (entity, year)
    years = y_panel.index.get_level_values('year').values

    # 1) Impute and lag X_panel
    X_imp = Imputer(method='drift').fit_transform(X_panel)
    X_lagged_panel = Lag(lags=[1,2,3]).fit_transform(X_imp)
    X_lagged = X_lagged_panel.values
    y = y_panel.values.ravel()

    # 2) Two-stage FS pipeline
    fs_pipe = Pipeline([
        ('all_rel', AllRelevantSelector(param_grid_stage1)),
        ('redund', RedundancyEliminator(param_grid=param_grid_stage2))
    ])

    # 3) Consensus feature selection across year-based folds using CutoffSplitter
    cutoffs = pd.to_datetime([f'{yr}-12-31' for yr in sorted(set(years))[:-1]])
    cv = CutoffSplitter(cutoffs=cutoffs, fh=1, window_length=None)
    # CutoffSplitter.split returns splits on y_panel's index

    fs_agg = FoldAggregator(pipeline=fs_pipe, cv=cv, n_jobs=4)
    fs_agg.fit(X_lagged, y)
    X_reduced = fs_agg.transform(X_lagged)

    # Convert back to DataFrame
    X_reduced_df = pd.DataFrame(X_reduced, index=X_panel.index,
                                columns=[f'feat_{i}' for i in range(X_reduced.shape[1])])

    # 4) Forecasting pipeline on reduced features
    panel_reducer = make_reduction(
        estimator=LGBMRegressor(), strategy='recursive', window_length=3, scitype='panel'
    )
    forecast_pipe = ForecastingPipeline([('forecaster', panel_reducer)])

    # 5) ForecastingGridSearchCV with same cutoffs
    param_grid = {
        'forecaster__estimator__n_estimators': [100, 200],
        'forecaster__estimator__learning_rate': [0.05, 0.1]
    }
    gscv = ForecastingGridSearchCV(
        forecaster=forecast_pipe, cv=cv, param_grid=param_grid, n_jobs=-1
    )
    gscv.fit(y_panel, X=X_reduced_df)
    print('Best params:', gscv.best_params_)
