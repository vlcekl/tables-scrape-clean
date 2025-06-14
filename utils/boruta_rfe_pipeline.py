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

class GroupTimeSeriesSplit:
    """Splits data by group labels in time order, sklearn-compatible."""
    def __init__(self, n_splits: int, groups: pd.Series):
        self.n_splits = n_splits
        self.groups = np.array(groups)
        self.unique_groups = np.unique(self.groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        grp = self.groups if groups is None else np.array(groups)
        unique = np.unique(grp)
        n = len(unique)
        test_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            start = (i + 1) * test_size
            end = start + test_size
            test_groups = unique[start:end]
            train_groups = unique[:start]
            train_idx = np.where(np.isin(grp, train_groups))[0]
            test_idx = np.where(np.isin(grp, test_groups))[0]
            yield train_idx, test_idx


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

def log_params(params: dict):
    mlflow.log_params(params)


def log_metric(name: str, value, step: int = None):
    mlflow.log_metric(name, value, step=step)


def compute_and_log_metrics(y_true, y_pred, y_proba, metrics: dict):
    for name, (fn, needs_proba) in metrics.items():
        val = fn(y_true, y_proba if needs_proba else y_pred)
        log_metric(name, val)


# Tuning helper
class Tuner(BaseEstimator, TransformerMixin):
    def __init__(
        self, estimator_name, problem_type, base_params,
        prefix, cv, random_state, n_trials=50, early_stop=50
    ):
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

    def transform(self, X: pd.DataFrame):
        return X

    def _tune(self, X, y):
        params = {**self.base_params}
        if self.estimator_name.lower() == 'lightgbm':
            params.setdefault('verbosity', -1)
            params['n_estimators'] = None
            tuner = LightGBMTunerCV(
                params, lgb.Dataset(X, label=y), folds=self.cv,
                early_stopping_rounds=self.early_stop,
                verbose_eval=False, seed=self.random_state
            )
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
                p['n_estimators'] = trial.suggest_int('n_estimators', 30, 1000)
                model = make_estimator('xgboost', self.problem_type, p, self.random_state)
                # simple CV
                scores = []
                for tr, val in self.cv.split(X, y):
                    model.fit(
                        X.iloc[tr], y.iloc[tr],
                        eval_set=[(X.iloc[val], y.iloc[val])],
                        early_stopping_rounds=self.early_stop,
                        callbacks=[XGBoostPruningCallback(trial, 'validation_0-logloss')],
                        verbose=False
                    )
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

        # prefix params and log
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
        boruta = BorutaPy(
            est, n_estimators='auto', max_iter=self.max_iter,
            random_state=self.random_state, n_jobs=self.n_jobs
        )
        boruta.fit(X.values, y.values)
        self.support_ = boruta.support_
        return self

    def transform(self, X: pd.DataFrame):
        return X.loc[:, X.columns[self.support_]]


class FBEDBackwardStep(BaseEstimator, TransformerMixin):
    def __init__(self, estimator_name, problem_type, cv, random_state, metric_name, metrics):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.cv = cv
        self.random_state = random_state
        self.metric_name = metric_name
        self.metrics = metrics

    def fit(self, X: pd.DataFrame, y: pd.Series=None):
        cols = list(X.columns)
        baseline = self._score_subset(X, y, cols)
        support = cols.copy()
        for col in cols:
            subset = [c for c in support if c != col]
            score = self._score_subset(X[subset], y, subset)
            if score >= baseline:
                support.remove(col)
                baseline = score
        self.support_ = [c in support for c in cols]
        return self

    def _score_subset(self, X_sub, y, cols):
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


# Build pipeline with clear flags

def build_pipeline(
    estimator_name, problem_type, base_params,
    *, n_splits=3, boruta_max_iter=50, random_state=42,
    groups: pd.Series, metric_name: str = 'accuracy', metrics: dict
):
    cv = GroupTimeSeriesSplit(n_splits, groups)
    steps = []
    steps.append(('tune_before', Tuner(estimator_name, problem_type, base_params, prefix='pre_', cv=cv, random_state=random_state)))
    steps.append(('boruta', BorutaStep(estimator_name, problem_type, random_state, boruta_max_iter, n_jobs=-1)))
    steps.append(('tune_after', Tuner(estimator_name, problem_type, base_params, prefix='mid_', cv=cv, random_state=random_state)))
    steps.append(('backward', FBEDBackwardStep(estimator_name, problem_type, cv, random_state, metric_name, metrics)))
    steps.append(('final', FinalEstimator(estimator_name, problem_type, random_state)))
    return Pipeline(steps)


# Parallel fold execution

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
    *, groups: pd.Series, estimator_name='randomforest',
    problem_type='binary', base_params=None,
    n_splits=5, boruta_max_iter=50,
    random_state=42, n_jobs=-1,
    metrics: dict, metric_name: str = 'accuracy'
):
    mlflow.set_experiment('FeatureSelection')
    outer = GroupTimeSeriesSplit(n_splits, groups)
    run_args = dict(
        estimator_name=estimator_name,
        problem_type=problem_type,
        base_params=base_params or {},
        n_splits=n_splits,
        boruta_max_iter=boruta_max_iter,
        random_state=random_state,
        groups=groups,
        metric_name=metric_name,
        metrics=metrics
    )
    tasks = [delayed(_run_fold)(i, tr, te, X, y, run_args)
             for i, (tr, te) in enumerate(outer.split(X, y))]
    results = Parallel(n_jobs=n_jobs)(tasks)
    return pd.DataFrame(results)


if __name__ == '__main__':

    # Example usage - Classification

    X_cls = pd.DataFrame(np.random.randn(1000, 20), columns=[f'f{i}' for i in range(20)])
    y_cls = pd.Series(np.random.randint(0, 4, 1000))
    groups = pd.Series(np.repeat(np.arange(10), 100))

    default_metrics = {
        'accuracy': (accuracy_score, False),
        'precision_macro': (lambda y,p: precision_score(y,p,average='macro'), False),
        'recall_macro': (lambda y,p: recall_score(y,p,average='macro'), False),
        'f1_macro': (lambda y,p: f1_score(y,p,average='macro'), False),
        'roc_auc_ovr': (lambda y,p: roc_auc_score(y,p,average='macro', multi_class='ovr'), True),
        'pr_auc_macro': (lambda y,p: average_precision_score(y,p,average='macro'), True)
    }
    df = run_parallel(
        X_cls, y_cls, groups=groups,
        estimator_name='randomforest', problem_type='multiclass',
        base_params={}, n_splits=5, boruta_max_iter=100,
        random_state=0, n_jobs=2,
        metrics=default_metrics, metric_name='accuracy'
    )
    print('CLASSIFICATION')
    print(df)

    # Example usage - Regression 

    X_reg = pd.DataFrame(np.random.randn(1000, 20), columns=[f'f{i}' for i in range(20)])
    y_reg = pd.Series(np.random.randn(1000))
    groups = pd.Series(np.repeat(np.arange(10), 100))

    default_metrics = {
        'mse': (mean_squared_error, False),
        'rmse': (lambda y,p: np.sqrt(mean_squared_error(y,p)), False),
        'mae': (mean_absolute_error, False),
        'r2': (r2_score, False)
    }
    df = run_parallel(
        X_reg, y_reg, groups=groups,
        estimator_name='randomforest', problem_type='regresssion',
        base_params={}, n_splits=5, boruta_max_iter=100,
        random_state=0, n_jobs=2,
        metrics=default_metrics, metric_name='mse'
    )
    print('REGRESSION')
    print(df)
