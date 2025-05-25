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


def _tune_with_optuna(X_train, y_train, estimator_name, params, random_state, cv, tune_n_estimators=True):
    """
    Tune estimator hyperparameters using Optuna.
    If tune_n_estimators is False, rely on early stopping to find best iteration.
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
                # tree complexity
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                # sampling
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                # regularization
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                # learning rate
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
            }
            # fixed large n_estimators if not tuning
            n_est = trial.suggest_int('n_estimators', 100, 1000) if tune_n_estimators else 1000
            model = XGBClassifier(
                **param,
                n_estimators=n_est,
                use_label_encoder=False,
                eval_metric='logloss',
                tree_method='hist',
                random_state=random_state
            )
            scores = []
            for train_ix, val_ix in cv.split(X_train):
                X_tr, X_val = X_train[train_ix], X_train[val_ix]
                y_tr, y_val = y_train[train_ix], y_train[val_ix]
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False,
                    callbacks=[XGBoostPruningCallback(trial, "validation_0-logloss")]
                )
                scores.append(model.score(X_val, y_val))
            if not tune_n_estimators:
                params['best_iteration'] = model.best_iteration
            return np.mean(scores)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=random_state),
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective, n_trials=50)
        best = study.best_params
        if not tune_n_estimators:
            best['n_estimators'] = 1000
        return best

    else:
        return params


def strategy_a_time_series_selection(X, y,
                                     estimator_name='randomforest',
                                     base_params=None,
                                     pre_tune=False,
                                     pre_tune_params=None,
                                     mid_tune=True,
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
    Strategy A feature selection with optional pre-Boruta tuning,
    mid-Boruta tuning before RFE, parallel outer folds,
    Boruta + blocked RFE + Optuna tuning.
    """
    mlflow.set_experiment('TimeSeriesFeatureSelection')
    tscv_outer = TimeSeriesSplit(n_splits=n_outer_splits, test_size=forecast_horizon)
    
    def process_fold(fold, train_idx, test_idx):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        with mlflow.start_run(run_name=f'fold_{fold}'):
            # log fold settings
            mlflow.log_params({
                'outer_fold': fold,
                'estimator': estimator_name,
                'pre_tune': pre_tune,
                'mid_tune': mid_tune,
                'n_outer_splits': n_outer_splits,
                'forecast_horizon': forecast_horizon,
                'n_inner_splits': n_inner_splits,
                'block_size': block_size,
                'rfe_step': rfe_step,
                'survival_threshold': survival_threshold,
                'boruta_max_iter': boruta_max_iter
            })
            # 1. Pre-Boruta tuning
            est_params = base_params or {}
            if pre_tune:
                cv_inner = TimeSeriesSplit(n_splits=n_inner_splits, test_size=block_size)
                tuned_pre = _tune_with_optuna(X_train, y_train, estimator_name, pre_tune_params or est_params,
                                             random_state, cv_inner, tune_n_estimators=False)
                mlflow.log_params({f'pre_{k}': v for k, v in tuned_pre.items()})
                est_params.update(tuned_pre)
            # 2. Boruta selection
            if estimator_name == 'randomforest':
                boruta_est = RandomForestClassifier(**est_params, n_jobs=-1, random_state=random_state)
            elif estimator_name == 'xgboost':
                boruta_est = XGBClassifier(**est_params, use_label_encoder=False, eval_metric='logloss',
                                           tree_method='hist', random_state=random_state)
            else:
                boruta_est = lgb.LGBMClassifier(**est_params, random_state=random_state)
            boruta_sel = BorutaPy(boruta_est,
                                  n_estimators=est_params.get('n_estimators', 'auto'),
                                  max_iter=boruta_max_iter,
                                  random_state=random_state)
            boruta_sel.fit(X_train, y_train)
            mask_boruta = boruta_sel.support_
            mlflow.log_metric('boruta_n_features', int(mask_boruta.sum()))
            X_boruta = X_train[:, mask_boruta]
            # 3. Mid-Boruta tuning before RFE
            if mid_tune:
                cv_mid = TimeSeriesSplit(n_splits=n_inner_splits, test_size=block_size)
                tuned_mid = _tune_with_optuna(X_boruta, y_train, estimator_name, est_params,
                                             random_state, cv_mid, tune_n_estimators=True)
                mlflow.log_params({f'mid_{k}': v for k, v in tuned_mid.items()})
                est_for_rfe = (RandomForestClassifier(**tuned_mid, n_jobs=-1, random_state=random_state)
                               if estimator_name=='randomforest' else (
                               XGBClassifier(**tuned_mid, use_label_encoder=False, eval_metric='logloss',
                                             tree_method='hist', random_state=random_state) if estimator_name=='xgboost' else (
                               lgb.LGBMClassifier(**tuned_mid, random_state=random_state))))
            else:
                est_for_rfe = boruta_est
            # 4. Blocked RFE with tuned estimator
            inner_tscv = TimeSeriesSplit(n_splits=n_inner_splits, test_size=block_size)
            block_supports = []
            for sub_tr, sub_val in inner_tscv.split(X_boruta):
                selector = RFE(est_for_rfe, n_features_to_select=1, step=rfe_step)
                selector.fit(X_boruta[sub_tr], y_train[sub_tr])
                block_supports.append(selector.support_)
            surv = np.mean(block_supports, axis=0)
            stable = surv >= survival_threshold
            idx_bor = np.where(mask_boruta)[0]
            selected_idx = idx_bor[stable]
            mlflow.log_metric('stable_n_features', int(len(selected_idx)))
            feat_names = [f'feat_{i}' for i in selected_idx]
            with open('features.txt','w') as f: f.write('\n'.join(feat_names))
            mlflow.log_artifact('features.txt')
            # 5. Post-elimination final training
            X_sel = X_train[:, selected_idx]
            cv_final = TimeSeriesSplit(n_splits=n_inner_splits, test_size=block_size)
            tuned_final = _tune_with_optuna(X_sel, y_train, estimator_name, (
                                           tuned_mid if mid_tune else est_params),
                                           random_state, cv_final, tune_n_estimators=True)
            mlflow.log_params({f'final_{k}': v for k, v in tuned_final.items()})
            final_est = (RandomForestClassifier(**tuned_final, n_jobs=-1, random_state=random_state)
                         if estimator_name=='randomforest' else (
                         XGBClassifier(**tuned_final, use_label_encoder=False, eval_metric='logloss',
                                       tree_method='hist', random_state=random_state) if estimator_name=='xgboost' else (
                         lgb.LGBMClassifier(**tuned_final, random_state=random_state))))
            final_est.fit(X_sel, y_train)
            score = final_est.score(X_test[:, selected_idx], y_test)
            mlflow.log_metric('test_score', float(score))
            return {'fold': fold, 'test_score': score}
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_fold)(i, tr, te) for i, (tr, te) in enumerate(tscv_outer.split(X))
    )
    return pd.DataFrame(results)


if __name__ == '__main__':
    X = np.random.randn(70000, 100)
    y = np.random.randint(0, 2, 70000)
    df = strategy_a_time_series_selection(
        X, y,
        estimator_name='xgboost',
        base_params={},
        pre_tune=True,
        pre_tune_params={'learning_rate':0.01},
        mid_tune=True,
        n_outer_splits=5,
        forecast_horizon=200,
        n_inner_splits=3,
        block_size=500,
        rfe_step=1,
        survival_threshold=0.7,
        boruta_max_iter=100,
        random_state=42,
        n_jobs=5
    )
    print(df)
