from sklearn.base import BaseEstimator, TransformerMixin
import optuna
import xgboost as xgb
from optuna.integration import XGBoostPruningCallback

class XGBoostTunerCV(BaseEstimator, TransformerMixin):
    """
    Single-study hyperparameter tuner for XGBoost mimicking LightGBMTunerCV interface,
    with optional pruning via XGBoostPruningCallback.

    Parameters
    ----------
    params : dict
        Initial parameters for XGBoost (e.g., objective, eval_metric).
    train_set : xgb.DMatrix, optional (default=None)
        Pre-constructed training DMatrix. If None, X and y must be passed to fit().
    folds : int or cross-validation splitter or list of (train_idx, val_idx), optional
        Custom folds. If int, interpreted as nfold for xgb.cv; if splitter,
        used to generate indices; if list, passed directly to xgb.cv via `folds`.
    n_trials : int, default=100
        Total number of trials for the optimization.
    time_budget : float, optional
        Maximum time (in seconds) for the optimization; stops when elapsed.
    random_state : int, default=42
        Random seed.
    early_stopping_rounds : int, default=50
        Early stopping rounds for the CV call.
    use_pruning : bool, default=False
        Whether to enable XGBoostPruningCallback for trial pruning.

    Attributes
    ----------
    best_params_ : dict
        Best parameters found (excluding num_boost_round).
    best_iteration_ : int
        Optimal number of boosting rounds from CV.
    booster_ : xgb.Booster
        Final booster trained on all training data.
    study_ : optuna.study.Study
        The Optuna study object.
    """
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

        # Optional time callback
        callbacks = []
        if self.time_budget is not None:
            callbacks.append(optuna.integration.MaxTimeCallback(self.time_budget))

        # Helper for CV: manual CV to support pruning
        def manual_cv(p, num_round, trial=None):
            # create folds
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
            # generate indices
            if hasattr(self.folds, 'split'):
                splits = list(self.folds.split(X, y))
            else:
                splits = self.folds
            results = []
            for fold_idx, (tr_idx, val_idx) in enumerate(splits):
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
            # wrap into DataFrame-like with mean
            import pandas as pd
            metric = p.get('eval_metric', 'rmse')
            return pd.DataFrame({f'test-{metric}-mean': [sum(results)/len(results)]})

        # Objective: tune joint parameters
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
            self.best_iteration_ = int(cvres[f'test-{p.get("eval_metric","rmse")}-mean'].index[-1])
            return cvres.iloc[-1, 0]

        direction = 'minimize' if self.params.get('objective', '').startswith('reg') else 'maximize'
        self.study_ = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        self.study_.optimize(objective, n_trials=self.n_trials, callbacks=callbacks)

        self.best_params_ = self.study_.best_params.copy()
        # ensure best_iteration exists
        if not hasattr(self, 'best_iteration_'):
            self.best_iteration_ = None
        # Train final booster
        self.booster_ = xgb.train(
            self.best_params_, dtrain,
            num_boost_round=self.best_iteration_
        )
        return self

    def run(self, X=None, y=None):
        return self.fit(X, y)

    def transform(self, X):
        return X

    def get_best_params(self):
        return self.best_params_

    def get_best_iteration(self):
        return self.best_iteration_

    def get_best_booster(self):
        return self.booster_


if __name__ == '__main__':
    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import KFold
    from sklearn.metrics import roc_auc_score

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    dtrain = xgb.DMatrix(X, label=y)

    tuner = XGBoostTunerCV(
        params={'objective': 'binary:logistic', 'eval_metric': 'auc'},
        train_set=dtrain,
        folds=kf,
        n_trials=100,
        time_budget=600,
        random_state=0,
        early_stopping_rounds=20,
        use_pruning=True
    )
    tuner.fit(X, y)
    print("Best params:", tuner.get_best_params())
    print("Best iteration:", tuner.get_best_iteration())
    preds = tuner.get_best_booster().predict(dtrain)
    print("AUC:", roc_auc_score(y, preds))
