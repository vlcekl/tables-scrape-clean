import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from boruta import BorutaPy
import mlflow
from joblib import Parallel, delayed
import optuna
from optuna.integration import LightGBMTunerCV, XGBoostPruningCallback
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.base import BaseEstimator, TransformerMixin
from BorutaShap import BorutaShap

class YearRollingSplit:
    """Rolling-year split (train on all years < y, validate on year y)"""
    def __init__(self, n_splits: int = 4, years: pd.Series = None):
        self.n_splits = n_splits
        self.years = years

    def get_n_splits(self, X=None, y=None):
        return self.n_splits

    def split(self, X, y=None, years: pd.Series = None):
        if years is None:
            split_years = np.array(self.years)
        else:
            split_years = np.array(years)

        unique_years = sorted(np.unique(split_years))
        i_min = max(1, len(unique_years) - self.n_splits)

        # iterate over years and return up to n_split latest splits
        for i, year in enumerate(unique_years):
            if i < i_min: continue
            train_idx = np.where(split_years < year)[0]
            val_idx = np.where(split_years == year)[0]
            yield train_idx, val_idx

class YearRandomTrainSplit:
    """
    Inner CV: keep the last year as validation, and do K random train/test splits within the earlier years.
    """
    def __init__(self, years: pd.Series, n_splits: int = 4, train_fraction: float = 0.75, random_state: int = 42):
        self.years = np.array(years)
        self.unique_years = sorted(np.unique(self.years))
        self.test_year = self.unique_years[-1]
        self.n_splits = n_splits
        self.train_fraction = train_fraction
        self.random_state = random_state

    def get_n_splits(self, X=None, y=None, years=None):
        return self.n_splits

    def split(self, X, y=None, years=None):
        # indices for train-validation subsampling
        train_idx_full = np.where(self.years < self.test_year)[0]
        test_idx = np.where(self.years == self.test_year)[0]
        rs = ShuffleSplit(
            n_splits=self.n_splits,
            train_size=self.train_fraction,
            random_state=self.random_state
        )
        for tr_idx, _ in rs.split(train_idx_full):
            tr = train_idx_full[tr_idx]
            yield tr, test_idx

# Common estimator factory

def make_estimator(name: str, problem: str, params: dict, random_state: int):
    params = params.copy()
    if 'random_state' not in params and 'seed' not in params:
        params['random_state'] = random_state
    if name.lower() == 'xgboost':
        Model = XGBClassifier if problem in ['binary', 'multiclass'] else XGBRegressor
        return Model(**params, tree_method='hist')
    if name.lower() == 'lightgbm':
        Model = lgb.LGBMClassifier if problem in ['binary', 'multiclass'] else lgb.LGBMRegressor
        return Model(**params)
    Model = RandomForestClassifier if problem in ['binary', 'multiclass'] else RandomForestRegressor
    params['n_jobs'] = params.get('n_jobs', -1)
    return Model(**params)

# Logging helpers

def log_params(params: dict): mlflow.log_params(params)

def log_metric(name: str, value, step: int = None): mlflow.log_metric(name, value, step=step)

def compute_and_log_metrics(y_true, y_pred, y_proba, metrics: dict):
    for name, (fn, needs_proba) in metrics.items():
        val = fn(y_true, y_proba if needs_proba else y_pred)
        log_metric(name, val)


# Tuning helper
class Tuner(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_name, problem_type, base_params,
                 prefix, cv, random_state, n_trials=50, early_stop=50):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.base_params = base_params or {}
        self.prefix = prefix
        self.cv = cv
        self.random_state = random_state
        self.n_trials = n_trials
        self.early_stop = early_stop

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        best = self._tune(X, y)
        self.best_params_ = best
        return self

    def transform(self, X: pd.DataFrame): return X

    def _tune(self, X, y):
        params = {**self.base_params}
        if self.estimator_name.lower() == 'lightgbm':
            params.setdefault('verbosity', -1)
            params['n_estimators'] = None
            tuner = LightGBMTunerCV(
                params, lgb.Dataset(X, label=y), folds=self.cv,
                early_stopping_rounds=self.early_stop,
                verbose_eval=False, seed=self.random_state)
            tuner.run()
            booster = tuner.get_best_booster()
            best = tuner.best_params
            best['n_estimators'] = booster.best_iteration
        elif self.estimator_name.lower() == 'xgboost':
            def objective(trial):
                p = {'verbosity':0, 'random_state':self.random_state}
                if self.problem_type == 'binary':
                    p.update({'objective':'binary:logistic','eval_metric':'logloss'})
                elif self.problem_type == 'multiclass':
                    p.update({'objective':'multi:softprob','eval_metric':'mlogloss','num_class':len(np.unique(y))})
                else:
                    p.update({'objective':'reg:squarederror','eval_metric':'rmse'})
                p['max_depth'] = trial.suggest_int('max_depth', 3, 15)
                p['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
                p['n_estimators'] = trial.suggest_int('n_estimators', 100, 1000)
                model = make_estimator('xgboost', self.problem_type, p, self.random_state)
                scores = []
                for tr, val in self.cv.split(X, y):
                    model.fit(
                        X.iloc[tr], y.iloc[tr],
                        eval_set=[(X.iloc[val], y.iloc[val])],
                        early_stopping_rounds=self.early_stop,
                        callbacks=[XGBoostPruningCallback(trial, 'validation_0-logloss')],
                        verbose=False)
                    pred = model.predict(X.iloc[val])
                    if self.problem_type in ['binary','multiclass']:
                        scores.append(np.mean(pred == y.iloc[val]))
                    else:
                        scores.append(-np.sqrt(((pred - y.iloc[val])**2).mean()))
                return np.mean(scores)
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
            study.optimize(objective, n_trials=self.n_trials)
            best = study.best_params
        else:
            best = params.copy()
        prefixed = {f"{self.prefix}{k}": v for k, v in best.items()}
        log_params(prefixed)
        return best


class BorutaStep(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_name, problem_type, random_state, max_iter, n_jobs=1):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_jobs = n_jobs

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        est = make_estimator(self.estimator_name, self.problem_type, {}, self.random_state)
        boruta = BorutaPy(est, n_estimators='auto', max_iter=self.max_iter,
                          random_state=self.random_state, n_jobs=self.n_jobs)
        boruta.fit(X.values, y.values)
        self.support_ = boruta.support_
        return self

    def transform(self, X: pd.DataFrame):
        return X.loc[:, X.columns[self.support_]]

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
            percentile=80, iterations=50, random_state=42
    )))
#    steps.append(('boruta', BorutaStep(estimator_name, problem_type,
#                                            random_state, boruta_max_iter, n_jobs=-1)))
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
        random_state=0, n_jobs=2,
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
        random_state=0, n_jobs=2,
        metrics=default_metrics, metric_name='mse',
        inner_splits=4, train_fraction=0.75
    )
    print('REGRESSION')
    print(df)