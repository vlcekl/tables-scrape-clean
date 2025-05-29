import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from BorutaShap import BorutaShap
import optuna
from optuna.integration import LightGBMTunerCV, XGBoostPruningCallback
from joblib import Parallel, delayed
import mlflow

# Logging helpers

def log_params(params: dict): mlflow.log_params(params)

def log_metric(name: str, value, step: int = None): mlflow.log_metric(name, value, step=step)

def compute_and_log_metrics(y_true, y_pred, y_proba, metrics: dict):
    for name, (fn, needs_proba) in metrics.items():
        val = fn(y_true, y_proba if needs_proba else y_pred)
        log_metric(name, val)

# Named metric functions for picklability

def precision_macro(y_true, y_pred):
    return precision_score(y_true, y_pred, average='macro')

def recall_macro(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

def f1_macro(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def roc_auc_ovr(y_true, y_proba):
    return roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')

def pr_auc_macro(y_true, y_proba):
    return average_precision_score(y_true, y_proba, average='macro')

# Custom CV splitters

class YearRollingSplit:
    """Rolling-year split (train on all years < y, validate on year y)"""
    def __init__(self, n_splits: int = 4, years: pd.Series = None):
        self.n_splits = n_splits
        self.years = np.array(years) if years is not None else None

    def get_n_splits(self, X=None, y=None):
        return sum(1 for _ in self.split(X, y))

    def split(self, X, y=None, years: pd.Series = None):
        split_years = np.array(years) if years is not None else self.years
        unique_years = sorted(np.unique(split_years))
        i_min = max(1, len(unique_years) - self.n_splits)
        for i, year in enumerate(unique_years):
            if i < i_min:
                continue
            train_idx = np.where(split_years < year)[0]
            val_idx = np.where(split_years == year)[0]
            yield train_idx, val_idx

class YearRandomTrainSplit:
    """
    Inner CV: keep the last year as validation, and do K random train/test splits within the earlier years.
    """
    def __init__(self, years: pd.Series, n_splits: int = 4, train_fraction: float = 0.75, random_state: int = 42):
        self.years = np.array(years)
        self.n_splits = n_splits
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, years: pd.Series = None):
        return self.n_splits

    def split(self, X, y=None, years: pd.Series = None):
        arr_years = np.array(years) if years is not None else self.years
        unique_years = sorted(np.unique(arr_years))
        test_year = unique_years[-1]
        train_idx_full = np.where(arr_years < test_year)[0]
        test_idx = np.where(arr_years == test_year)[0]
        rs = ShuffleSplit(
            n_splits=self.n_splits,
            train_size=self.train_fraction,
            random_state=self.random_state
        )
        for tr_idx, _ in rs.split(train_idx_full):
            tr = train_idx_full[tr_idx]
            yield tr, test_idx

# Estimator factory with defensive copying

def make_estimator(name: str, problem: str, params: dict, random_state: int):
    params_copy = params.copy()
    if 'random_state' not in params_copy and 'seed' not in params_copy:
        params_copy['random_state'] = random_state
    if name.lower() == 'xgboost':
        Model = xgb.XGBClassifier if problem in ['binary', 'multiclass'] else xgb.XGBRegressor
        return Model(**params_copy, tree_method='hist')
    if name.lower() == 'lightgbm':
        Model = lgb.LGBMClassifier if problem in ['binary', 'multiclass'] else lgb.LGBMRegressor
        return Model(**params_copy)
    Model = RandomForestClassifier if problem in ['binary', 'multiclass'] else RandomForestRegressor
    params_copy['n_jobs'] = params_copy.get('n_jobs', -1)
    return Model(**params_copy)

# XGBoostTunerCV corrected

class XGBoostTunerCV(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        params,
        train_set=None,
        folds=None,
        n_trials=100,
        time_budget=None,
        random_state=42,
        early_stopping_rounds=50,
        use_pruning=False
    ):
        self.params = params.copy()
        self.train_set = train_set
        self.folds = folds
        self.n_trials = n_trials
        self.time_budget = time_budget
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.use_pruning = use_pruning

    def fit(self, X=None, y=None):
        # Prepare training data
        if self.train_set is None:
            if X is None or y is None:
                raise ValueError("X and y must be provided if train_set is not set.")
            dtrain = xgb.DMatrix(X, label=y)
        else:
            dtrain = self.train_set

        base_params = {k: v for k, v in self.params.items() if k not in ['eta', 'num_boost_round']}
        callbacks = []
        if self.time_budget is not None:
            callbacks.append(optuna.integration.MaxTimeCallback(self.time_budget))

        def manual_cv(p, num_round, trial=None):
            if isinstance(self.folds, int) or self.folds is None:
                nfold = self.folds or 3
                cvres = xgb.cv(
                    p, dtrain,
                    nfold=nfold,
                    num_boost_round=num_round,
                    early_stopping_rounds=self.early_stopping_rounds,
                    seed=self.random_state,
                    metrics=p.get('eval_metric', 'rmse')
                )
                return cvres
            if hasattr(self.folds, 'split'):
                splits = list(self.folds.split(X, y))
            else:
                splits = self.folds
            results = []
            for tr_idx, val_idx in splits:
                dtr = xgb.DMatrix(X.iloc[tr_idx], label=y.iloc[tr_idx])
                dval = xgb.DMatrix(X.iloc[val_idx], label=y.iloc[val_idx])
                local_callbacks = []
                if self.use_pruning and trial is not None:
                    local_callbacks.append(
                        XGBoostPruningCallback(trial, f'validation-{p.get("eval_metric","rmse")}-mean')
                    )
                bst = xgb.train(
                    p,
                    dtr,
                    num_boost_round=num_round,
                    evals=[(dval, 'validation')],
                    early_stopping_rounds=self.early_stopping_rounds,
                    callbacks=local_callbacks,
                    verbose_eval=False
                )
                results.append(bst.best_score)
            import pandas as pd
            metric = p.get('eval_metric', 'rmse')
            return pd.DataFrame({f'test-{metric}-mean': [sum(results)/len(results)]})

        def objective(trial):
            p = base_params.copy()
            p['max_depth'] = trial.suggest_int('max_depth', 3, 12)
            p['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)
            p['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            p['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            p['reg_alpha'] = trial.suggest_loguniform('reg_alpha', 1e-8, 10.0)
            p['reg_lambda'] = trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
            p['eta'] = trial.suggest_loguniform('eta', 1e-3, 1e-1)
            num_round = trial.suggest_int('num_boost_round', 30, 1000)
            cvres = manual_cv(p, num_round, trial if self.use_pruning else None)
            self.best_iteration_ = int(cvres.index[-1])
            return cvres.iloc[-1, 0]

        direction = 'minimize' if self.params.get('objective', '').startswith('reg') else 'maximize'
        self.study_ = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        self.study_.optimize(objective, n_trials=self.n_trials, callbacks=callbacks)

        self.best_params_ = self.study_.best_params.copy()
        if not hasattr(self, 'best_iteration_'):
            self.best_iteration_ = None
        self.booster_ = xgb.train(
            {**self.params, **self.best_params_},
            dtrain,
            num_boost_round=self.best_iteration_
        )
        return self

    def transform(self, X):
        return X

    def get_best_params(self):
        return self.study_.best_params.copy()

    def get_best_iteration(self):
        return self.best_iteration_

    def get_best_booster(self):
        return self.booster_

# Tuner corrected to propagate parameters

class Tuner(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_name, problem_type, base_params,
                 prefix, cv, random_state, n_trials=50, early_stop=50):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.base_params = base_params.copy()
        self.prefix = prefix
        self.cv = cv
        self.random_state = random_state
        self.n_trials = n_trials
        self.early_stop = early_stop

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        best = self._tune(X, y)
        # update shared base_params so downstream estimators pick them up
        self.base_params.update(best)
        return self

    def transform(self, X: pd.DataFrame):
        return X

    def _tune(self, X, y):
        params = self.base_params.copy()
        if self.estimator_name.lower() == 'lightgbm':
            params.setdefault('verbosity', -1)
            params['n_estimators'] = None
            tuner = LightGBMTunerCV(
                params, lgb.Dataset(X, label=y), folds=self.cv,
                early_stopping_rounds=self.early_stop,
                verbose_eval=False, seed=self.random_state)
            tuner.run()
            booster = tuner.get_best_booster()
            best = tuner.best_params.copy()
            best['n_estimators'] = booster.best_iteration
        elif self.estimator_name.lower() == 'xgboost':
            tuner = XGBoostTunerCV(
                params, xgb.DMatrix(X, label=y), folds=self.cv,
                early_stopping_rounds=self.early_stop,
                n_trials=self.n_trials, time_budget=None,
                random_state=self.random_state,
                use_pruning=False)
            tuner.fit()
            best = tuner.get_best_params()
            best['n_estimators'] = tuner.get_best_iteration()
        else:
            best = params.copy()
        return best


class BorutaSHAPStep(BaseEstimator, TransformerMixin):
    """
    All-relevant feature selection via Boruta-Shap.

    Parameters
    ----------
    model : str or estimator, default='lightgbm'
        Which base learner to explain with SHAP.  Accepts:
        - a string: 'lightgbm', 'xgboost', 'random_forest'
        - or any fitted tree-based estimator instance (must implement .fit/.predict)

    importance_measure : {'shap','gain','perm'}, default='shap'
        Which importance to use for testing.  'shap' is recommended for both
        RandomForest and GBMs; 'gain' will fall back to built-in tree gain;
        'perm' will use permutation importances.

    classification : bool, default=True
        Whether the problem is classification (True) or regression (False).

    percentile : int in [1,100], default=100
        When sampling data internally to approximate SHAP, what percentile of
        data to keep on each iteration (higher => more samples => slower but
        more precise).

    iterations : int, default=100
        Number of Boruta iterations to run (i.e. shadow-feature comparisons).

    random_state : int, RandomState instance or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    support_ : 1d boolean array of length n_features
        Mask of features accepted as “important” by BorutaShap.

    feature_importances_ : pandas.Series
        Mean absolute SHAP (or chosen) importances for each original feature.
    """

    def __init__(
        self,
        model: Any = 'lightgbm',
        importance_measure: str = 'shap',
        classification: bool = True,
        percentile: int = 100,
        iterations: int = 100,
        random_state: Optional[int] = None
    ):
        self.model = model
        self.importance_measure = importance_measure
        self.classification = classification
        self.percentile = percentile
        self.iterations = iterations
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the BorutaShap selector on X, y.
        """
        # initialize BorutaShap
        self._selector = BorutaShap(
            model=self.model,
            importance_measure=self.importance_measure,
            classification=self.classification,
            percentile=self.percentile,
            iterations=self.iterations,
            random_state=self.random_state
        )

        # BorutaShap expects numpy or pd.DataFrame
        self._selector.fit(X=X, y=y, normalize=True)

        # accepted_ is list of selected feature names
        accepted = set(self._selector.accepted)
        self.support_ = [col in accepted for col in X.columns]

        # store importances for diagnostics
        importances = dict(zip(X.columns, self._selector.shap_values_mean if hasattr(self._selector, 'shap_values_mean') else []))
        self.feature_importances_ = pd.Series(importances).loc[X.columns]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce X to only the accepted features.
        """
        return X.loc[:, self.support_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class FBEDBackwardStep(BaseEstimator, TransformerMixin):
    """Backward elimination with inner random-train / fixed-year validation CV"""
    def __init__(self, estimator_name, problem_type, cv, random_state, metric_name, metrics):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.cv = cv
        self.random_state = random_state
        self.metric_name = metric_name
        self.metrics = metrics

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        cols = list(X.columns)
        baseline = self._score_subset(X, y)
        support = cols.copy()
        for col in cols:
            subset = [c for c in support if c != col]
            score = self._score_subset(X[subset], y)
            if score >= baseline:
                support.remove(col)
                baseline = score
        self.support_ = [c in support for c in cols]
        return self

    def _score_subset(self, X_sub, y):
        model = make_estimator(self.estimator_name, self.problem_type, {}, self.random_state)
        scores = []
        for tr, val in self.cv.split(X_sub, y):
            model.fit(X_sub.iloc[tr], y.iloc[tr])
            pred = model.predict(X_sub.iloc[val])
            scores.append(self.metrics[self.metric_name][0](y.iloc[val], pred))
        return np.mean(scores)

    def transform(self, X: pd.DataFrame):
        return X.loc[:, X.columns[self.support_]]


class FinalEstimator(BaseEstimator):
    def __init__(self, estimator_name, problem_type, random_state):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        self.model_ = make_estimator(self.estimator_name, self.problem_type, {}, self.random_state)
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(X)


# Build pipeline using YearKFoldTrainSplit for inner CV

def build_pipeline(
    estimator_name, problem_type, base_params,
    *, tune_before_fs=False, tune_after_fs=True,
    boruta_max_iter=50, random_state=42,
    years: pd.Series, metric_name: str = 'accuracy', metrics: dict,
    inner_splits: int = 3, train_fraction: float = 0.8
):
    inner_cv = YearRandomTrainSplit(years, n_splits=inner_splits,
                                     train_fraction=train_fraction,
                                     random_state=random_state)
    steps = []
    if tune_before_fs:
        steps.append(('tune_before', Tuner(estimator_name, problem_type, base_params,
                                           prefix='pre_', cv=inner_cv,
                                           random_state=random_state)))
    steps.append(('borutashap', BorutaSHAPStep(
            model='lightgbm', importance_measure='shap', classification=(problem_type in ['binary','multiclass']),
            percentile=80, iterations=boruta_max_iter, random_state=random_state
    )))
    if tune_after_fs:
        steps.append(('tune_after', Tuner(estimator_name, problem_type, base_params,
                                          prefix='mid_', cv=inner_cv,
                                          random_state=random_state)))
    steps.append(('backward', FBEDBackwardStep(estimator_name, problem_type,
                                               inner_cv, random_state,
                                               metric_name, metrics)))
    steps.append(('final', FinalEstimator(estimator_name, problem_type, random_state)))
    return Pipeline(steps)


# Parallel fold execution with outer time split

def _run_fold(i, tr, te, X, y, run_args):
    try:
        with mlflow.start_run(run_name=f'fold_{i}'):
            mlflow.set_tag('fold', i)
            pipe = build_pipeline(**run_args)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            preds = pipe.predict(X.iloc[te])
            probas = pipe.predict_proba(X.iloc[te])
            compute_and_log_metrics(y.iloc[te], preds, probas, run_args['metrics'])
            res = {'fold': i}
            for name, (fn, needs) in run_args['metrics'].items():
                res[name] = fn(y.iloc[te], probas if needs else preds)
        return res
    except Exception as e:
        mlflow.log_param(f'fold_{i}_error', str(e))
        return {'fold': i, 'error': str(e)}

def run_parallel(
    X: pd.DataFrame, y: pd.Series,
    *, years: pd.Series, estimator_name='randomforest',
    problem_type='binary', base_params=None,
    tune_before_fs=False, tune_after_fs=True,
    n_splits=5, boruta_max_iter=50,
    random_state=42, n_jobs=-1,
    metrics: dict, metric_name: str = 'accuracy',
    inner_splits: int = 3, train_fraction: float = 0.8
):
    mlflow.set_experiment('FeatureSelection')
    outer_cv = YearRollingSplit(n_splits, years=years)
    run_args = dict(
        estimator_name=estimator_name,
        problem_type=problem_type,
        base_params=base_params or {},
        tune_before_fs=tune_before_fs,
        tune_after_fs=tune_after_fs,
        boruta_max_iter=boruta_max_iter,
        random_state=random_state,
        years=years,
        metric_name=metric_name,
        metrics=metrics,
        inner_splits=inner_splits,
        train_fraction=train_fraction
    )
    tasks = [delayed(_run_fold)(i, tr, te, X, y, run_args)
             for i, (tr, te) in enumerate(outer_cv.split(X, y))]
    results = Parallel(n_jobs=n_jobs)(tasks)
    return pd.DataFrame(results)


if __name__ == '__main__':

    # Example usage - Classification

    X_cls = pd.DataFrame(np.random.randn(200, 20), columns=[f'f{i}' for i in range(20)])
    y_cls = pd.Series(np.random.randint(0, 2, 200))
    years = pd.Series(np.repeat(np.arange(2015, 2025), 20))

    default_metrics = {
        'accuracy': (accuracy_score, False),
        'precision_macro': (lambda y,p: precision_score(y,p,average='macro'), False),
        'recall_macro': (lambda y,p: recall_score(y,p,average='macro'), False),
        'f1_macro': (lambda y,p: f1_score(y,p,average='macro'), False),
        'roc_auc_ovr': (lambda y,p: roc_auc_score(y,p,average='macro', multi_class='ovr'), True),
        'pr_auc_macro': (lambda y,p: average_precision_score(y,p,average='macro'), True)
    }
    df = run_parallel(
        X_cls, y_cls, years=years,
        estimator_name='randomforest', problem_type='multiclass',
        base_params={}, n_splits=5, boruta_max_iter=100,
        tune_before_fs=False, tune_after_fs=True,
        random_state=0, n_jobs=8,
        metrics=default_metrics, metric_name='accuracy',
        inner_splits=4, train_fraction=0.75
    )
    print('CLASSIFICATION')
    print(df)

    # Example usage - Regression 

    X_reg = pd.DataFrame(np.random.randn(200, 20), columns=[f'f{i}' for i in range(20)])
    y_reg = pd.Series(np.random.randn(200))
    years = pd.Series(np.repeat(np.arange(2015, 2025), 20))

    default_metrics = {
        'mse': (mean_squared_error, False),
        'rmse': (lambda y,p: np.sqrt(mean_squared_error(y,p)), False),
        'mae': (mean_absolute_error, False),
        'r2': (r2_score, False)
    }
    df = run_parallel(
        X_reg, y_reg, years=years,
        estimator_name='randomforest', problem_type='regression',
        base_params={}, n_splits=5, boruta_max_iter=100,
        tune_before_fs=False, tune_after_fs=True,
        random_state=0, n_jobs=8,
        metrics=default_metrics, metric_name='mse',
        inner_splits=4, train_fraction=0.75
    )
    print('REGRESSION')
    print(df)