import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from sklearn.model_selection import TimeSeriesSplit
import mlflow
from joblib import Parallel, delayed
import optuna
from optuna.integration import LightGBMTunerCV, XGBoostPruningCallback


def make_estimator(name: str, problem: str, params: dict, random_state: int):
    """Factory to construct estimator by name and problem type."""
    if name.lower() == 'xgboost':
        Model = XGBClassifier if problem in ['binary','multiclass'] else XGBRegressor
        return Model(**params, tree_method='hist', random_state=random_state)
    if name.lower() == 'lightgbm':
        Model = lgb.LGBMClassifier if problem in ['binary','multiclass'] else lgb.LGBMRegressor
        return Model(**params, random_state=random_state)
    Model = RandomForestClassifier if problem in ['binary','multiclass'] else RandomForestRegressor
    return Model(**params, n_jobs=-1, random_state=random_state)


def get_objective_settings(name: str, problem: str, y=None):
    """Return objective and metric settings for tuning."""
    settings = {}
    if name.lower() == 'xgboost':
        if problem == 'binary':
            settings = {'objective':'binary:logistic','eval_metric':'logloss'}
        elif problem == 'multiclass' and y is not None:
            settings = {'objective':'multi:softprob','eval_metric':'mlogloss','num_class':len(np.unique(y))}
        else:
            settings = {'objective':'reg:squarederror','eval_metric':'rmse'}
    else:
        if problem == 'binary':
            settings['objective'] = 'binary'
        elif problem == 'multiclass' and y is not None:
            settings = {'objective':'multiclass','num_class':len(np.unique(y))}
        else:
            settings['objective'] = 'regression'
    return settings


def log_params(params: dict, prefix: str = ""):
    mlflow.log_params({f"{prefix}{k}": v for k, v in params.items()})


def log_metric(name: str, value):
    mlflow.log_metric(name, value)


class Tuner(BaseEstimator, TransformerMixin):
    """Generic tuner transformer using Optuna for classification/regression."""
    def __init__(self, estimator_name, problem_type, base_params, prefix,
                 tune_n_estimators, cv, random_state,
                 n_trials=50, early_stop=50):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.base_params = base_params or {}
        self.prefix = prefix
        self.tune_n_estimators = tune_n_estimators
        self.cv = cv
        self.random_state = random_state
        self.n_trials = n_trials
        self.early_stop = early_stop

    def _xgb_objective(self, trial, X, y):
        p = {'verbosity':0, 'seed':self.random_state}
        p.update(get_objective_settings('xgboost', self.problem_type, y))
        # suggest hyperparameters
        p['max_depth'] = trial.suggest_int('max_depth', 3, 15)
        p['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)
        p['gamma'] = trial.suggest_float('gamma', 0.0, 5.0)
        p['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
        p['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)
        p['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
        p['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
        p['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        n_est = trial.suggest_int('n_estimators', 100, 1000) if self.tune_n_estimators else self.base_params.get('n_estimators', 1000)
        model = make_estimator('xgboost', self.problem_type, {**p, 'n_estimators': n_est}, self.random_state)
        scores = []
        for tr, val in self.cv.split(X):
            model.fit(
                X[tr], y[tr],
                eval_set=[(X[val], y[val])],
                early_stopping_rounds=self.early_stop,
                callbacks=[XGBoostPruningCallback(trial, f"validation_0-{'mlogloss' if self.problem_type=='multiclass' else 'logloss' if self.problem_type=='binary' else 'rmse'}")],
                verbose=False
            )
            pred = model.predict(X[val])
            score = np.mean(pred == y[val]) if self.problem_type in ['binary','multiclass'] else -np.sqrt(((pred - y[val])**2).mean())
            scores.append(score)
        return np.mean(scores)

    def _tune(self, X, y):
        params = {**self.base_params, **getattr(self, 'prev_params', {})}
        if self.estimator_name.lower() == 'lightgbm':
            settings = get_objective_settings('lightgbm', self.problem_type, y)
            params.setdefault('verbosity', -1)
            params.update(settings)
            train_set = lgb.Dataset(X, label=y)
            tuner = LightGBMTunerCV(
                params, train_set, folds=self.cv,
                early_stopping_rounds=self.early_stop,
                verbose_eval=False, seed=self.random_state
            )
            tuner.run()
            best = tuner.best_params
        elif self.estimator_name.lower() == 'xgboost':
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(lambda t: self._xgb_objective(t, X, y), n_trials=self.n_trials)
            best = study.best_params
            if not self.tune_n_estimators:
                best.setdefault('n_estimators', params.get('n_estimators', 1000))
        else:
            best = params.copy()
        log_params(best, prefix=self.prefix)
        return best

    def fit(self, X, y, **fit_params):
        self.prev_params = fit_params.get('prev_params', {})
        tuned = self._tune(X, y)
        self.params_ = {**self.prev_params, **tuned}
        return self

    def transform(self, X):
        return X


class BorutaStep(BaseEstimator, TransformerMixin):
    """Boruta feature selection step with MLflow logging."""
    def __init__(self, estimator_name, problem_type, random_state, max_iter):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X, y, **fit_params):
        params = fit_params.get('prev_params', {})
        est = make_estimator(self.estimator_name, self.problem_type, params, self.random_state)
        boruta = BorutaPy(
            est,
            n_estimators=params.get('n_estimators', 'auto'),
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        boruta.fit(X, y)
        self.support_ = boruta.support_
        log_metric('boruta_n_features', int(self.support_.sum()))
        return self

    def transform(self, X):
        return X.iloc[:, self.support_] if hasattr(X, 'iloc') else X[:, self.support_]


class RFEPruner(BaseEstimator, TransformerMixin):
    """Redundancy pruning via blocked RFE with MLflow logging."""
    def __init__(self, estimator_name, problem_type, random_state, cv, step, survival_threshold):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.random_state = random_state
        self.cv = cv
        self.step = step
        self.survival_threshold = survival_threshold

    def fit(self, X, y, **fit_params):
        params = fit_params.get('prev_params', {})
        supports = []
        for tr, val in self.cv.split(X):
            est = make_estimator(self.estimator_name, self.problem_type, params, self.random_state)
            sel = RFE(est, n_features_to_select=1, step=self.step)
            sel.fit(X[tr], y[tr])
            supports.append(sel.support_)
        surv = np.mean(supports, axis=0)
        self.support_ = surv >= self.survival_threshold
        log_metric('stable_n_features', int(self.support_.sum()))
        return self

    def transform(self, X):
        return X.iloc[:, self.support_] if hasattr(X, 'iloc') else X[:, self.support_]


class FinalEstimator(BaseEstimator):
    """Final estimator step for classification/regression."""
    def __init__(self, estimator_name, problem_type, random_state):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.random_state = random_state

    def fit(self, X, y, **fit_params):
        params = fit_params.get('prev_params', {})
        self.model_ = make_estimator(self.estimator_name, self.problem_type, params, self.random_state)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def build_pipeline(estimator_name, problem_type, base_params,
                   pre_tune=False, mid_tune=True,
                   n_inner_splits=3, rfe_step=1, survival_threshold=0.7,
                   boruta_max_iter=50, random_state=42):
    cv = TimeSeriesSplit(n_splits=n_inner_splits)
    steps = []
    if pre_tune:
        steps.append(('pre_tune', Tuner(estimator_name, problem_type, base_params,
                                         'pre_', False, cv, random_state)))
    steps.append(('boruta', BorutaStep(estimator_name, problem_type, random_state, boruta_max_iter)))
    if mid_tune:
        steps.append(('mid_tune', Tuner(estimator_name, problem_type, base_params,
                                         'mid_', True, cv, random_state)))
    steps.append(('rfe', RFEPruner(estimator_name, problem_type, random_state,
                                    cv, rfe_step, survival_threshold)))
    steps.append(('final_tune', Tuner(estimator_name, problem_type, base_params,
                                       'final_', True, cv, random_state)))
    steps.append(('estimator', FinalEstimator(estimator_name, problem_type, random_state)))
    return Pipeline(steps)


def run_parallel(
    X, y,
    estimator_name='randomforest',
    problem_type='classification',
    base_params=None,
    pre_tune=False,
    mid_tune=True,
    n_outer_splits=5,
    forecast_horizon=50,
    n_inner_splits=3,
    rfe_step=1,
    survival_threshold=0.7,
    boruta_max_iter=50,
    random_state=42,
    n_jobs=-1,
    metrics=None
):
    """
    Run time-series feature selection in parallel across outer CV folds.
    """
    mlflow.set_experiment('TimeSeriesFeatureSelection')
    tscv_outer = TimeSeriesSplit(n_splits=n_outer_splits, test_size=forecast_horizon)
    if metrics is None:
        if problem_type in ['binary', 'multiclass']:
            metrics = {'accuracy': lambda y, p: np.mean(p == y)}
        else:
            metrics = {'neg_rmse': lambda y, p: -np.sqrt(((p - y)**2).mean())}

    def _evaluate_fold(fold, tr_idx, val_idx):
        pipe = build_pipeline(
            estimator_name, problem_type, base_params,
            pre_tune, mid_tune, n_inner_splits,
            rfe_step, survival_threshold,
            boruta_max_iter, random_state
        )
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        with mlflow.start_run(run_name=f'fold_{fold}'):
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)
            results = {}
            for name, fn in metrics.items():
                val = fn(y_val, preds)
                log_metric(name, val)
                results[name] = val
            return {'fold': fold, **results}

    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_fold)(i, tr, te)
        for i, (tr, te) in enumerate(tscv_outer.split(X))
    )
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Example data
    X_cls = np.random.randn(1000, 20)
    y_cls = np.random.randint(0, 4, 1000)

    metrics_cls = {
        'accuracy': lambda y, p: np.mean(p == y),
        'f1_macro': lambda y, p: None  # placeholder for actual f1 computation
    }

    # XGBoost multiclass example
    df_xgb = run_parallel(
        X_cls, y_cls,
        estimator_name='xgboost',
        problem_type='multiclass',
        base_params={},
        pre_tune=True, mid_tune=True,
        n_outer_splits=3, forecast_horizon=50,
        n_inner_splits=2, rfe_step=1,
        survival_threshold=0.5, boruta_max_iter=20,
        random_state=0, n_jobs=2,
        metrics=metrics_cls
    )
    print("XGBoost multiclass results:\n", df_xgb)

    # RandomForest multiclass example
    df_rf = run_parallel(
        X_cls, y_cls,
        estimator_name='randomforest',
        problem_type='multiclass',
        base_params={'n_estimators': 100},
        pre_tune=False, mid_tune=False,
        n_outer_splits=3, forecast_horizon=50,
        n_inner_splits=2, rfe_step=1,
        survival_threshold=0.5, boruta_max_iter=20,
        random_state=0, n_jobs=2,
        metrics=metrics_cls
    )
    print("RandomForest multiclass results:\n", df_rf)

    # Regression example
    X_reg = np.random.randn(1000, 20)
    y_reg = np.random.randn(1000)
    df_reg = run_parallel(
        X_reg, y_reg,
        estimator_name='lightgbm',
        problem_type='regression',
        base_params={},
        pre_tune=True, mid_tune=True,
        n_outer_splits=3, forecast_horizon=50,
        n_inner_splits=2, rfe_step=1,
        survival_threshold=0.5, boruta_max_iter=20,
        random_state=0, n_jobs=2
    )
    print("Regression results:\n", df_reg)
