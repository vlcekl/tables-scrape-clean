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

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn

# Enable autologging
mlflow.sklearn.autolog()

class AllRelevantSelector(BaseEstimator, TransformerMixin):
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def fit(self, X, y):
        # Stage 1: hyperparameter search
        gs = GridSearchCV(
            estimator=LGBMClassifier(),
            param_grid=self.param_grid,
            cv=3
        )
        with mlflow.start_run(nested=True):
            gs.fit(X, y)
            # Log best params explicitly if needed
            mlflow.log_params(gs.best_params_)
        # Feature selection
        selector = SelectFromModel(gs.best_estimator_, threshold='median')
        selector.fit(X, y)
        self.support_ = selector.get_support()
        self.best_params_ = gs.best_params_
        return self

    def transform(self, X):
        return X[:, self.support_]

class RedundancyEliminator(BaseEstimator, TransformerMixin):
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def fit(self, X, y):
        # Stage 2: hyperparameter re-tuning on reduced features
        gs = GridSearchCV(
            estimator=LGBMClassifier(),
            param_grid=self.param_grid,
            cv=3
        )
        with mlflow.start_run(nested=True):
            gs.fit(X, y)
            mlflow.log_params(gs.best_params_)
        # Redundancy elimination (e.g., remove highly correlated)
        X_sel = X.copy()
        corr = np.abs(np.corrcoef(X_sel, rowvar=False))
        to_remove = set()
        for i in range(corr.shape[0]):
            for j in range(i):
                if corr[i, j] > 0.9:
                    to_remove.add(i)
        mask = np.array([i not in to_remove for i in range(X_sel.shape[1])])
        self.support_ = mask
        self.best_params_ = gs.best_params_
        return self

    def transform(self, X):
        return X[:, self.support_]

if __name__ == '__main__':


    # Assemble pipeline
    param_grid_stage1 = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
    param_grid_stage2 = {'n_estimators': [30, 60], 'learning_rate': [0.01, 0.1]}

    pipe = Pipeline([
        ('all_rel', AllRelevantSelector(param_grid_stage1)),
        ('redund', RedundancyEliminator(param_grid_stage2)),
        ('final', LGBMClassifier())  # Final fit also autologged
    ])

    # Fit pipeline (this logs nested runs for each stage and final model)
    X_train = np.random.randn(200, 20)
    y_train = np.random.randint(0, 2, 200)
    with mlflow.start_run(run_name="two_stage_fs"):
        pipe.fit(X_train, y_train)

    # Extract results
    print("Stage 1 best params:", pipe.named_steps['all_rel'].best_params_)
    print("Stage 1 selected features:", pipe.named_steps['all_rel'].support_.sum())
    print("Stage 2 best params:", pipe.named_steps['redund'].best_params_)
    print("Stage 2 selected features:", pipe.named_steps['redund'].support_.sum())

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier

class FoldAggregator(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline, cv=None, n_jobs=1):
        self.pipeline = pipeline
        self.cv = cv or TimeSeriesSplit(n_splits=5)
        self.n_jobs = n_jobs

    def _fit_one(self, X, y, train_idx, test_idx, fold_id):
        """Fit a cloned pipeline on one fold and extract artifacts."""
        pipe = clone(self.pipeline)
        X_tr, y_tr = X[train_idx], y[train_idx]
        pipe.fit(X_tr, y_tr)
        # Example: two-stage selectors named 'all_rel' and 'redund'
        mask1 = pipe.named_steps['all_rel'].support_
        params1 = pipe.named_steps['all_rel'].best_params_
        mask2 = pipe.named_steps['redund'].support_
        params2 = pipe.named_steps['redund'].best_params_
        # Could compute score on test set if desired
        return {'fold': fold_id,
                'mask1': mask1, 'params1': params1,
                'mask2': mask2, 'params2': params2}

    def fit(self, X, y=None):
        # Prepare indices
        splits = list(self.cv.split(X, y))
        # Parallel fit
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one)(X, y, tr, ts, i)
            for i, (tr, ts) in enumerate(splits)
        )
        # Aggregate
        self.masks1_ = np.vstack([r['mask1'] for r in results])
        self.params1_ = [r['params1'] for r in results]
        self.masks2_ = np.vstack([r['mask2'] for r in results])
        self.params2_ = [r['params2'] for r in results]
        # Consensus mask: features selected in every fold
        self.consensus_mask1_ = self.masks1_.all(axis=0)
        self.consensus_mask2_ = self.masks2_.all(axis=0)
        return self

    def transform(self, X):
        # Apply consensus feature reduction (stage 2 mask supersedes stage1)
        mask = self.consensus_mask2_
        return X[:, mask]

    def get_fold_summary(self):
        """Return a DataFrame summarizing per-fold parameters."""
        df = pd.DataFrame({
            'fold': list(range(len(self.params1_))),
            'stage1_params': self.params1_,
            'stage2_params': self.params2_
        })
        return df


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
        ('final', LGBMClassifier())  # Final fit also autologged
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
