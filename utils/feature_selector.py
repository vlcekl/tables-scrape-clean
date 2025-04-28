"""
Feature Engineering and Selection Pipeline with MLflow Tracking.

This script:
 1. Loads configuration from S3 (cfg dict).
 2. Merges CSV tables and splits into train/validation by a feature threshold.
 3. Engineers lagged and interaction features.
 4. Pre-filters features via LightGBM built-in importance.
 5. Applies a hybrid forward-backward greedy selection wrapper,
    tracking each step and final results in MLflow.
"""
import json
import boto3
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

# ----- Data Loading and Splitting -----
def load_and_split_data(cfg: dict):
    """
    Read CSV tables from S3 as per cfg['tables'], merge them via cfg['joins'],
    and split into train/validation based on cfg['split']['column'] and 'value'.
    Returns X_train, X_val, y_train, y_val.
    """
    # Load each table with specified features
    tables = {}
    for tbl in cfg['tables']:
        df = pd.read_csv(f"s3://{tbl['bucket']}/{tbl['key']}")
        tables[tbl['name']] = df[tbl['features']]

    # Merge tables sequentially
    df_main = None
    for j in cfg['joins']:
        left, right, on, how = j['left'], j['right'], j['on'], j.get('how', 'inner')
        if df_main is None:
            df_main = tables[left].merge(tables[right], on=on, how=how)
        else:
            df_main = df_main.merge(tables[right], on=on, how=how)

    # Split into train/validation by threshold
    col, val = cfg['split']['column'], cfg['split']['value']
    if col not in df_main.columns:
        raise KeyError(f"Split column '{col}' not found in merged data")
    mask = df_main[col] <= val
    df_train = df_main[mask]
    df_val   = df_main[~mask]

    # Extract feature matrices and labels
    features = cfg['features']
    label    = cfg['label']
    X_train = df_train[features]
    y_train = df_train[label]
    X_val   = df_val[features]
    y_val   = df_val[label]
    return X_train, X_val, y_train, y_val

# Map operation keywords to functions
OPS = {
    'ratio': lambda x, y: x / (y + 1e-6),  # avoid divide-by-zero
    'diff':  lambda x, y: x - y,
    'prod':  lambda x, y: x * y,
    'sum':   lambda x, y: x + y,
}

# ----- Feature Engineering Helpers -----
def add_lag_features(df: pd.DataFrame, lag_specs: list) -> pd.DataFrame:
    """
    Add lagged versions of columns.

    lag_specs: list of (feature_name, lag_period) tuples.
    """
    df_copy = df.copy()
    for feat, lag in lag_specs:
        if feat not in df_copy.columns:
            raise KeyError(f"Feature '{feat}' not found for lag.")
        df_copy[f"{feat}_lag{lag}"] = df_copy[feat].shift(lag)
    df_copy.fillna(0, inplace=True)
    return df_copy


def add_interactions(df: pd.DataFrame, interactions: list) -> pd.DataFrame:
    """
    Add pairwise interaction features.

    interactions: list of (new_name, feat_i, feat_j, op_key) tuples.
    """
    df_copy = df.copy()
    for new_name, fi, fj, op_key in interactions:
        if op_key not in OPS:
            raise ValueError(f"Unknown op '{op_key}' for interaction '{new_name}'.")
        if fi not in df_copy or fj not in df_copy:
            raise KeyError(f"Features '{fi}' or '{fj}' missing for '{new_name}'.")
        df_copy[new_name] = OPS[op_key](df_copy[fi].values, df_copy[fj].values)
    return df_copy

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
                     K: int, n_jobs: int,
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

    # Forward phase: add up to (K - len(always_keep)) features
    candidates = [f for f in prefilter if f not in selected]
    max_add = K - len(always_keep)
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
def engineer_features(cfg: dict,
                      X_train: pd.DataFrame,
                      X_val: pd.DataFrame,
                      y_train: pd.Series,
                      y_val: pd.Series) -> list:
    """
    Orchestrate full feature engineering and selection.

    cfg keys used:
      - prev_selected, prev_near_misses, interactions, lag_specs,
        K, n_jobs, tables, joins, split, features, label
    Returns sorted final feature list.
    """
    # Extract parameters
    always_keep = set(cfg['prev_selected'])
    near_misses = set(cfg['prev_near_misses'])
    interactions = cfg['interactions']
    lag_specs = cfg.get('lag_specs', [])
    K = cfg['K']; n_jobs = cfg['n_jobs']

    # Build initial candidate set
    all_cols = set(X_train.columns)
    candidates = always_keep | near_misses | (all_cols - always_keep - near_misses)

    # Prepare train/val feature matrices
    X_tr = X_train[list(candidates)]; X_va = X_val[list(candidates)]

    # Step 1: lag features
    if lag_specs:
        X_tr = add_lag_features(X_tr, lag_specs)
        X_va = add_lag_features(X_va, lag_specs)

    # Step 2: interaction features
    X_tr = add_interactions(X_tr, interactions)
    X_va = add_interactions(X_va, interactions)

    # Step 3: pre-filter by built-in importance
    dtrain = lgb.Dataset(X_tr, label=y_train)
    lgb_params = {
        'objective':'binary', 'metric':'auc', 'learning_rate':0.05,
        'verbose':-1, 'num_threads': n_jobs,
        'num_boost_round':100, 'early_stopping_rounds':10
    }
    bst = lgb.train(
        {k: v for k, v in lgb_params.items() if k in ['objective','metric','learning_rate','verbose','num_threads']},
        dtrain,
        num_boost_round=cfg['K']
    )
    imp = bst.feature_importance(importance_type='gain')
    feats = bst.feature_name()
    ranked = [f for _, f in sorted(zip(imp, feats), reverse=True)]
    prefilter = list(always_keep) + ranked[:max(0, K - len(always_keep))]

    # Step 4: run hybrid selection under MLflow run
    with mlflow.start_run():
        # Log entire config and data sizes
        mlflow.log_dict(cfg, 'config.json')
        mlflow.log_param('train_rows', X_tr.shape[0])
        mlflow.log_param('val_rows',   X_va.shape[0])

        final = hybrid_selection(
            prefilter,
            always_keep,
            X_tr, y_train,
            X_va, y_val,
            K, n_jobs,
            lgb_params
        )

    return sorted(final)


if __name__ == '__main__':
    # 1. Load configuration from S3
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket='ml-config', Key='run_config.json')
    cfg = json.load(obj['Body'])

    # 2. Read & merge tables, split into train/val
    X_train, X_val, y_train, y_val = load_and_split_data(cfg)

    # 3. Run engineering pipeline
    final_features = engineer_features(cfg, X_train, X_val, y_train, y_val)
    print('Final features:', final_features)
