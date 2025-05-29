import pandas as pd
import xgboost as xgb
import optuna
from optuna.integration import XGBoostPruningCallback


class XGBoostTunerCV:
    """
    An Optuna-based tuner for XGBoost (CV + pruning), inspired by LightGBMTunerCV.

    Methods mirror LightGBMTunerCV: fit/run, get_best_params, get_best_iteration, get_best_booster.

    Parameters
    ----------
    params : dict
        Base XGBoost parameters (without 'eta' or 'num_boost_round').
    folds : int, cross-validator or list of (train_idx, valid_idx), default=3
        Number of CV folds, or custom splitter, or list of index tuples.
    n_trials : int, default=100
        Number of Optuna trials.
    time_budget : float or None
        Time budget in seconds for tuning. None means no limit.
    seed : int, default=42
        Seed for reproducibility.
    early_stopping_rounds : int, default=50
        Rounds for early stopping in CV.
    use_pruning : bool, default=False
        Whether to enable Optuna pruning via XGBoostPruningCallback.
    """
    def __init__(
        self,
        params: dict,
        folds=None,
        n_trials: int = 100,
        time_budget: float = None,
        seed: int = 42,
        early_stopping_rounds: int = 50,
        use_pruning: bool = False
    ):
        self.params = params.copy()
        self.folds = folds if folds is not None else 3
        self.n_trials = n_trials
        self.time_budget = time_budget
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.use_pruning = use_pruning

    def _prepare_dmatrix(self, X, y):
        if hasattr(self, 'train_set') and self.train_set is not None:
            return self.train_set
        return xgb.DMatrix(X, label=y)

    def _iterate_splits(self, X, y):
        if isinstance(self.folds, int):
            return None  # Use built-in cv
        if hasattr(self.folds, 'split'):
            cv = self.folds.split(X, y)
        else:
            cv = self.folds
        for train_idx, valid_idx in cv:
            dtr = xgb.DMatrix(X.iloc[train_idx], label=y.iloc[train_idx])
            dval = xgb.DMatrix(X.iloc[valid_idx], label=y.iloc[valid_idx])
            yield dtr, dval

    def _run_cv(self, params_trial, num_boost_round, X, y, trial=None):
        metric = params_trial.get('eval_metric', 'rmse')
        if isinstance(self.folds, int):
            return xgb.cv(
                params_trial,
                self._prepare_dmatrix(X, y),
                num_boost_round=num_boost_round,
                nfold=self.folds,
                seed=self.seed,
                metrics=[metric],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False
            )
        scores = []
        for dtrain_split, dvalid_split in self._iterate_splits(X, y):
            cbs = []
            if self.use_pruning and trial is not None:
                cbs.append(XGBoostPruningCallback(trial, f'validation-{metric}-mean'))
            bst = xgb.train(
                params_trial,
                dtrain_split,
                num_boost_round=num_boost_round,
                evals=[(dvalid_split, 'validation')],
                early_stopping_rounds=self.early_stopping_rounds,
                callbacks=cbs,
                verbose_eval=False
            )
            scores.append(bst.best_score)
        return pd.DataFrame({f'test-{metric}-mean': [sum(scores) / len(scores)]})

    def fit(self, X=None, y=None, train_set=None):
        """
        Tune hyperparameters with Optuna and train final booster.
        Alias: run()
        """
        if train_set is not None:
            self.train_set = train_set
        self._X, self._y = X, y
        base_params = {k: v for k, v in self.params.items() if k not in ['eta', 'num_boost_round']}

        def objective(trial: optuna.trial.Trial) -> float:
            params_trial = base_params.copy()
            params_trial.update({
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
                'eta': trial.suggest_loguniform('eta', 1e-3, 1e-1)
            })
            num_round = trial.suggest_int('num_boost_round', 30, 1000)
            cvres = self._run_cv(params_trial, num_round, X, y, trial if self.use_pruning else None)
            self.best_iteration = int(cvres.index[-1])
            return float(cvres.iloc[-1, 0])

        direction = 'minimize' if self.params.get('objective', '').startswith('reg') else 'maximize'
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        if self.time_budget:
            study.optimize(objective, n_trials=self.n_trials, timeout=self.time_budget)
        else:
            study.optimize(objective, n_trials=self.n_trials)

        self.study = study
        self.best_params = study.best_params.copy()

        dtrain = self._prepare_dmatrix(X, y)
        final_params = {**self.params, **self.best_params}
        self.booster = xgb.train(
            final_params,
            dtrain,
            num_boost_round=self.best_iteration
        )
        return self

    # alias for LightGBMTunerCV compatibility
    run = fit

    def get_best_params(self) -> dict:
        return self.best_params.copy()

    def get_best_iteration(self) -> int:
        return self.best_iteration

    def get_booster(self) -> xgb.Booster:
        return self.booster

    # alias method
    def get_best_booster(self) -> xgb.Booster:
        return self.get_booster()
