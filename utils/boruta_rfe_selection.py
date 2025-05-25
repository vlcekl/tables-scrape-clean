import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from sklearn.model_selection import TimeSeriesSplit
import mlflow
from joblib import Parallel, delayed
import optuna
from optuna.integration import LightGBMTunerCV, XGBoostPruningCallback


def _tune_with_optuna(X_train, y_train, estimator_name, params, random_state, cv):
    """
    Tune estimator hyperparameters after Boruta selection using Optuna.
    Integrates pruning callback for XGBoost trials.
    """
    if estimator_name.lower() == 'lightgbm':
        train_set = lgb.Dataset(X_train, label=y_train)
        tuner = LightGBMTunerCV(
            params,
            train_set,
            folds=cv,
            early_stopping_rounds=50,
            verbose_eval=False,
            seed=random_state
        )
        tuner.run()
        return tuner.best_params

    elif estimator_name.lower() == 'xgboost':
        def objective(trial):
            param = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'verbosity': 0,
                'seed': random_state,
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0)
            }
            model = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss', random_state=random_state)
            scores = []
            for train_ix, val_ix in cv.split(X_train):
                X_tr, X_val = X_train[train_ix], X_train[val_ix]
                y_tr, y_val = y_train[train_ix], y_train[val_ix]
                # Use pruning callback to stop unpromising trials early
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=30,
                    verbose=False,
                    callbacks=[XGBoostPruningCallback(trial, "validation_0-logloss")]
                )
                scores.append(model.score(X_val, y_val))
            return np.mean(scores)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=random_state),
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective, n_trials=20)
        return study.best_params

    else:
        # Default: return provided params for RandomForest
        return params


def strategy_a_time_series_selection(X, y,
                                     estimator_name='randomforest',
                                     base_params=None,
                                     n_outer_splits=5,
                                     forecast_horizon=50,
                                     n_inner_splits=3,
                                     block_size=100,
                                     rfe_step=1,
                                     survival_threshold=0.7,
                                     boruta_max_iter=50,
                                     random_state=42,
                                     n_jobs=-1):
    """
    Strategy A feature selection with parallel outer folds, Boruta + blocked RFE + Optuna tuning.
    """
    mlflow.set_experiment('TimeSeriesFeatureSelection')
    tscv_outer = TimeSeriesSplit(n_splits=n_outer_splits, test_size=forecast_horizon)
    
    def process_fold(fold, train_idx, test_idx):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        with mlflow.start_run(run_name=f'fold_{fold}'):
            # Log fold and settings
            mlflow.log_params({
                'outer_fold': fold,
                'estimator': estimator_name,
                'n_outer_splits': n_outer_splits,
                'forecast_horizon': forecast_horizon,
                'n_inner_splits': n_inner_splits,
                'block_size': block_size,
                'rfe_step': rfe_step,
                'survival_threshold': survival_threshold,
                'boruta_max_iter': boruta_max_iter
            })
            # 1. Boruta
            base_est = RandomForestClassifier(n_jobs=-1, random_state=random_state) if estimator_name=='randomforest' else (
                       XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state) if estimator_name=='xgboost' else (
                       lgb.LGBMClassifier(random_state=random_state)))
            boruta_sel = BorutaPy(base_est, n_estimators='auto', max_iter=boruta_max_iter, random_state=random_state)
            boruta_sel.fit(X_train, y_train)
            mask_boruta = boruta_sel.support_
            mlflow.log_metric('boruta_n_features', int(mask_boruta.sum()))
            X_boruta = X_train[:, mask_boruta]
            
            # 2. Blocked RFE
            inner_tscv = TimeSeriesSplit(n_splits=n_inner_splits, test_size=block_size)
            block_supports = []
            for sub_tr, sub_val in inner_tscv.split(X_boruta):
                X_sub, y_sub = X_boruta[sub_tr], y_train[sub_tr]
                selector = RFE(base_est, n_features_to_select=1, step=rfe_step)
                selector.fit(X_sub, y_sub)
                block_supports.append(selector.support_)
            surv = np.mean(block_supports, axis=0)
            stable = surv >= survival_threshold
            idx_bor = np.where(mask_boruta)[0]
            selected_idx = idx_bor[stable]
            mlflow.log_metric('stable_n_features', int(len(selected_idx)))
            feat_names = [f'feat_{i}' for i in selected_idx]
            with open('features.txt','w') as f: f.write('\n'.join(feat_names))
            mlflow.log_artifact('features.txt')
            
            # 3. Hyperparameter tuning
            X_sel = X_train[:, selected_idx]
            cv_inner = TimeSeriesSplit(n_splits=n_inner_splits, test_size=block_size)
            tuned_params = _tune_with_optuna(X_sel, y_train, estimator_name, base_params or {}, random_state, cv_inner)
            mlflow.log_params(tuned_params)
            
            # 4. Final training
            final_est = (RandomForestClassifier(**tuned_params, n_jobs=-1, random_state=random_state)
                         if estimator_name=='randomforest' else (
                         XGBClassifier(**tuned_params, use_label_encoder=False, eval_metric='logloss', random_state=random_state) if estimator_name=='xgboost' else (
                         lgb.LGBMClassifier(**tuned_params, random_state=random_state))))
            final_est.fit(X_sel, y_train)
            X_test_sel = X_test[:, selected_idx]
            score = final_est.score(X_test_sel, y_test)
            mlflow.log_metric('test_score', float(score))
            return {'fold': fold, 'test_score': score}
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_fold)(i, tr, te) for i, (tr, te) in enumerate(tscv_outer.split(X))
    )
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Synthetic example
    X = np.random.randn(1000, 50)
    y = np.random.randint(0, 2, 1000)
    df = strategy_a_time_series_selection(
        X, y,
        estimator_name='xgboost',
        base_params={},
        n_outer_splits=5,
        forecast_horizon=100,
        n_inner_splits=3,
        block_size=100,
        rfe_step=1,
        survival_threshold=0.7,
        boruta_max_iter=50,
        random_state=42,
        n_jobs=5
    )
    print(df)
