"""
Feature Engineering and Selection Pipeline with MLflow Tracking.

This script:
 1. Loads configuration from S3 (cfg dict).
 2. Select LightGBM hyperparameters suitable for the dataset
 3. Pre-filters features via LightGBM built-in importance.
 4. Applies a hybrid forward-backward greedy selection wrapper,
    tracking each step and final results in MLflow.
"""
import json
import boto3
import pandas as pd
import lightgbm as lgb
from optuna.integration.lightgbm import LightGBMTunerCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
import mlflow


# ----- LightGBM parameter selection/tuning -----
def optimize_lgb_params(X_train, y_train, cfg, n_splits=5, time_budget=600):
    """
    Optimize LightGBM hyperparameters using Optuna's LightGBMTunerCV or use provided parameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        cfg (dict): Configuration dictionary containing 'optimize_lgb' flag and 'lgb_params'.
        n_splits (int): Number of cross-validation folds.
        time_budget (int): Time budget for optimization in seconds.

    Returns:
        dict: Best hyperparameters found by Optuna or provided parameters.
    """
    # Check if optimization is enabled in the configuration
    if cfg.get('optimize_lgb', True):
        # Prepare dataset
        dtrain = lgb.Dataset(X_train, label=y_train)

        # Initialize KFold cross-validation
        kf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Initialize the tuner
        tuner = LightGBMTunerCV(
            cfg['lgb_params'],
            dtrain,
            folds=kf.split(X_train, y_train, X_train[cfg['split']['column']]),
            time_budget=time_budget,
            verbosity=-1,
        )

        # Run the tuner
        tuner.run()

        # Return the best parameters
        return tuner.best_params
    else:
        # Return provided parameters if optimization is skipped
        return cfg.get('lgb_params', {})

# ----- Data Loading and Splitting -----
def load_and_split_data(cfg: dict):
    """
    Read a CSV table from S3 and split it into train/validation based on cfg['split']['column'] and 'value'.
    Returns X_train, X_val, y_train, y_val.
    """
    # Load table with specified features
    df = pd.read_csv(f"s3://{tbl['bucket']}/{tbl['key']}")

    # Split into train/validation by threshold
    col, val = cfg['split']['column'], cfg['split']['value']
    if col not in df.columns:
        raise KeyError(f"Split column '{col}' not found in data")
    mask = df[col] <= val
    df_train = df[mask]
    df_val   = df[~mask]

    # Extract feature matrices and labels
    features = cfg['features']
    label    = cfg['label']
    X_train = df_train[features]
    y_train = df_train[label]
    X_val   = df_val[features]
    y_val   = df_val[label]
    return X_train, X_val, y_train, y_val

# ----- Model Evaluation -----
def evaluate_native(feats: list, X_tr: pd.DataFrame, y_tr: pd.Series,
                    X_va: pd.DataFrame, y_va: pd.Series,
                    lgb_params: dict) -> float:
    """
    Train native LightGBM on given features and return AUC on validation.
    """
    dtrain = lgb.Dataset(X_tr[feats], label=y_tr)
    dval = lgb.Dataset(X_va[feats], label=y_va, reference=dtrain)
    bst = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=lgb_params['num_boost_round'],
        valid_sets=[dval],
        early_stopping_rounds=lgb_params['early_stopping_rounds'],
        verbose_eval=False
    )
    preds = bst.predict(X_va[feats], num_iteration=bst.best_iteration)
    return roc_auc_score(y_va, preds)

# ----- Hybrid Forward-Backward Selection -----
def hybrid_selection(prefilter: list,
                     always_keep: set,
                     X_tr: pd.DataFrame, y_tr: pd.Series,
                     X_va: pd.DataFrame, y_va: pd.Series,
                     K_max: int, n_jobs: int,
                     lgb_params: dict) -> list:
    """
    Greedy wrapper: forward addition then backward removal.
    Logs each step's selected set and score to MLflow.
    """
    # Log initial feature sets
    mlflow.log_param('prefilter_features', json.dumps(prefilter))
    mlflow.log_param('always_keep', json.dumps(list(always_keep)))

    selected = list(always_keep)
    # baseline score with only always_keep
    baseline = evaluate_native(selected, X_tr, y_tr, X_va, y_va, lgb_params)
    mlflow.log_metric('baseline_score', baseline, step=0)

    # Forward phase: add up to (K_max - len(always_keep)) features
    candidates = [f for f in prefilter if f not in selected]
    max_add = K_max - len(always_keep)
    step = 1
    for _ in range(max_add):
        # parallel evaluation of adding each candidate
        def eval_add(f):
            return (f, evaluate_native(selected + [f], X_tr, y_tr, X_va, y_va, lgb_params))
        results = Parallel(n_jobs=n_jobs)(delayed(eval_add)(f) for f in candidates)
        best_f, best_score = max(results, key=lambda x: x[1])
        if best_score > baseline:
            selected.append(best_f)
            candidates.remove(best_f)
            baseline = best_score
            mlflow.log_metric('forward_score', best_score, step=step)
            mlflow.log_param(f'selected_after_forward_{step}', json.dumps(selected))
            step += 1
        else:
            break

    # Backward phase: remove features not in always_keep
    removable = [f for f in selected if f not in always_keep]
    step = 1
    for _ in range(len(removable)):
        def eval_rem(f):
            subset = [x for x in selected if x != f]
            return (f, evaluate_native(subset, X_tr, y_tr, X_va, y_va, lgb_params))
        results = Parallel(n_jobs=n_jobs)(delayed(eval_rem)(f) for f in removable)
        rem_f, rem_score = max(results, key=lambda x: x[1])
        if rem_score >= baseline:
            selected.remove(rem_f)
            removable.remove(rem_f)
            baseline = rem_score
            mlflow.log_metric('backward_score', rem_score, step=step)
            mlflow.log_param(f'selected_after_backward_{step}', json.dumps(selected))
            step += 1
        else:
            break

    mlflow.log_param('final_selection', json.dumps(selected))
    return selected

# ----- Main Pipeline -----
def select_features(cfg: dict,
                    X_train: pd.DataFrame,
                    X_val: pd.DataFrame,
                    y_train: pd.Series,
                    y_val: pd.Series) -> list:
    """
    Orchestrate full feature engineering and selection.

    cfg keys used:
      - prev_selected, prev_near_misses, interactions, lag_specs,
        K_max, n_jobs, tables, joins, split, features, label
    Returns sorted final feature list.
    """
    # Extract parameters
    always_keep = set(cfg['prev_selected'])
    K_max = cfg['K_max']
    n_jobs = cfg['n_jobs']

    dtrain = lgb.Dataset(X_train, label=y_train)

    # Step 1: Choose LightGBM parameters by using
    # (i) defaults with (early stopping and best_iteration)
    # (ii) parameters provided in lgb_params item of cfg
    # (iii) optuna.integration.lightgbm.LightGBMTunerCV to find best params on the full dataset
    #TODO: modify the code below and write function that returns lgb_params and implements the above choices
        # Step 1: Optimize LightGBM hyperparameters or use provided parameters

    lgb_params = optimize_lgb_params(X_train, y_train, cfg)

    # Step 2: pre-filter by built-in importance
    bst = lgb.train(lgb_params, dtrain)

    imp = bst.feature_importance(importance_type='gain')
    feats = bst.feature_name()
    ranked = [f for _, f in sorted(zip(imp, feats), reverse=True)]
    prefilter = list(always_keep) + ranked[:max(0, K_max - len(always_keep))]

    # Step 3: run hybrid selection under MLflow run
    with mlflow.start_run():
        # Log entire config and data sizes
        mlflow.log_dict(cfg, 'config.json')
        mlflow.log_param('train_rows', X_train.shape[0])
        mlflow.log_param('val_rows',   X_val.shape[0])

        final = hybrid_selection(
            prefilter,
            always_keep,
            X_train, y_train,
            X_val, y_val,
            K_max, n_jobs,
            lgb_params
        )

    return sorted(final)


if __name__ == '__main__':
    # 1. Load configuration from S3
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket='ml-config', Key='feature_select_config.json')
    cfg = json.load(obj['Body'])

    # 2. Read & merge tables, split into train/val
    X_train, X_val, y_train, y_val = load_and_split_data(cfg)

    # 3. Run feature select pipeline
    final_features = select_features(cfg, X_train, X_val, y_train, y_val)
    print('Final features:', final_features)
