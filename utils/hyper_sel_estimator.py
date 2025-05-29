import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
from BorutaShap import BorutaShap
import optuna
from optuna.integration import LightGBMTunerCV, XGBoostPruningCallback

from sklearn.model_selection import ShuffleSplit

# assume make_estimator and XGBoostTunerCV are defined as before

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
    # fallback to random forest
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    Model = RandomForestClassifier if problem in ['binary', 'multiclass'] else RandomForestRegressor
    params_copy['n_jobs'] = params_copy.get('n_jobs', -1)
    return Model(**params_copy)


class HyperSelEstimator(BaseEstimator):
    """
    Single estimator combining hyperparameter tuning and feature selection (via Boruta-Shap).

    Parameters
    ----------
    estimator_name : {'lightgbm','xgboost','randomforest'}
    problem_type : {'binary','multiclass','regression'}
    base_params : dict
        Hyperparameter search space defaults.
    cv : cross-validation splitter for tuning
    n_trials : int
        Number of Optuna trials (for XGBoost only).
    boruta_params : dict
        Parameters passed to BorutaShap.
    random_state : int
    """
    def __init__(
        self,
        estimator_name: str,
        problem_type: str,
        base_params: dict,
        cv,
        n_trials: int = 50,
        boruta_params: dict = None,
        random_state: int = 42
    ):
        self.estimator_name = estimator_name
        self.problem_type = problem_type
        self.base_params = base_params.copy()
        self.cv = cv
        self.n_trials = n_trials
        self.boruta_params = boruta_params or {}
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # 1. Hyperparameter tuning
        params = self.base_params.copy()
        if self.estimator_name.lower() == 'lightgbm':
            params.setdefault('verbosity', -1)
            params['n_estimators'] = None
            tuner = LightGBMTunerCV(
                params,
                lgb.Dataset(X, label=y),
                folds=self.cv,
                early_stopping_rounds=self.boruta_params.get('early_stopping_rounds', 50),
                verbose_eval=False,
                seed=self.random_state
            )
            tuner.run()
            best_params = tuner.best_params.copy()
            best_params['n_estimators'] = tuner.get_best_booster().best_iteration
        elif self.estimator_name.lower() == 'xgboost':
            # simple Optuna tuning using XGBoostTunerCV
            tuner = XGBoostTunerCV(
                self.base_params,
                train_set=xgb.DMatrix(X, label=y),
                folds=self.cv,
                n_trials=self.n_trials,
                random_state=self.random_state,
                early_stopping_rounds=self.boruta_params.get('early_stopping_rounds', 50),
                use_pruning=False
            )
            tuner.fit(X, y)
            best_params = tuner.get_best_params()
            best_params['n_estimators'] = tuner.get_best_iteration()
        else:
            # no tuning for random forest
            best_params = params

        self.best_params_ = best_params

        # 2. Feature selection via Boruta-Shap
        model_for_shap = make_estimator(
            self.estimator_name,
            self.problem_type,
            self.best_params_,
            self.random_state
        )

        boruta = BorutaShap(
            model=model_for_shap,
            **self.boruta_params,
            random_state=self.random_state
        )
        boruta.fit(X=X, y=y, normalize=True)
        accepted = set(boruta.accepted)
        self.support_ = [col in accepted for col in X.columns]

        # 3. Final model fit on selected features
        self.model_ = make_estimator(
            self.estimator_name,
            self.problem_type,
            self.best_params_,
            self.random_state
        )
        X_sel = X.loc[:, self.support_]
        self.model_.fit(X_sel, y)
        return self

    def predict(self, X: pd.DataFrame):
        X_sel = X.loc[:, self.support_]
        return self.model_.predict(X_sel)

    def predict_proba(self, X: pd.DataFrame):
        X_sel = X.loc[:, self.support_]
        return self.model_.predict_proba(X_sel)

    def get_support(self):
        return np.array(self.support_)

    def get_best_params(self):
        return self.best_params_.copy()


# Example usage
if __name__ == '__main__':
    # generate synthetic data
    X = pd.DataFrame(np.random.randn(200, 20), columns=[f'f{i}' for i in range(20)])
    y = pd.Series(np.random.randint(0, 2, 200))

    # define inner CV
    years = pd.Series(np.repeat(np.arange(2015, 2025), 20))
    from sklearn.model_selection import KFold
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)

    clf = HyperSelEstimator(
        estimator_name='lightgbm',
        problem_type='binary',
        base_params={'learning_rate': 0.05},
        cv=inner_cv,
        n_trials=30,
        boruta_params={'importance_measure': 'shap', 'iterations': 50},
        random_state=0
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    print("Selected features:", np.where(clf.get_support())[0])
