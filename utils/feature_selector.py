import json
import boto3
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# ---------- Operation Functions ----------
OPS = {
    'ratio': lambda x, y: x / (y + 1e-6),
    'diff':  lambda x, y: x - y,
    'prod':  lambda x, y: x * y,
    'sum':   lambda x, y: x + y,
}

# ---------- Feature Engineering ----------
def add_interactions(df, interactions):
    """
    Create specified pairwise interaction features using OPS mapping.
    interactions: list of tuples (new_name, feat_i, feat_j, op_key)
    """
    df = df.copy()
    for name, fi, fj, op_key in interactions:
        op = OPS.get(op_key)
        if op is None:
            raise ValueError(f"Unknown operation '{op_key}' for interaction {name}")
        if fi not in df.columns or fj not in df.columns:
            raise KeyError(f"Interaction features '{fi}' or '{fj}' not in DataFrame columns")
        df[name] = op(df[fi].values, df[fj].values)
    return df

# ---------- Built-in Importance ----------
def get_builtin_importance(booster, feature_names, importance_type='gain'):
    importances = booster.feature_importance(importance_type=importance_type)
    df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    return df.sort_values('importance', ascending=False).reset_index(drop=True)

# ---------- Permutation Importance ----------
def get_permutation_importance(booster, X_valid, y_valid, features,
                               n_rounds=5, random_state=42):
    baseline = roc_auc_score(y_valid, booster.predict(X_valid[features]))
    imp_dict = {f: [] for f in features}
    rng = np.random.RandomState(random_state)

    for _ in range(n_rounds):
        for f in features:
            Xp = X_valid.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            perm = roc_auc_score(y_valid, booster.predict(Xp[features]))
            imp_dict[f].append(baseline - perm)

    records = [
        {'feature': f,
         'importance_mean': np.mean(vals),
         'importance_std': np.std(vals)}
        for f, vals in imp_dict.items()
    ]
    imp_df = pd.DataFrame.from_records(records)
    return imp_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)

# ---------- Main Workflow ----------
def main(X_train, X_val, y_train, y_val,
         prev_selected, prev_near_misses,
         interactions, K=100, perm_rounds=10):
    """
    Perform feature selection with interactions.
    Returns sorted list of final features.
    """
    always_keep = set(prev_selected)
    near_misses = set(prev_near_misses)
    fresh_cols  = set(X_train.columns)
    new_features = fresh_cols - always_keep - near_misses

    # Expand datasets with only specified interactions
    candidates = always_keep | near_misses | new_features
    X_tr_all = X_train[list(candidates)]
    X_va_all = X_val[list(candidates)]
    X_tr_exp = add_interactions(X_tr_all, interactions)
    X_va_exp = add_interactions(X_va_all, interactions)

    # Train baseline booster
    dtrain = lgb.Dataset(X_tr_exp, label=y_train)
    params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.05, 'verbose': -1}
    bst = lgb.train(params, dtrain, num_boost_round=100)

    # Built-in importance pre-filter
    b_imp = get_builtin_importance(bst, dtrain.feature_name(), 'gain')
    rest = [f for f in b_imp['feature'] if f not in always_keep]
    top_rest = rest[:max(0, K - len(always_keep))]
    prefilter = list(always_keep) + top_rest

    # Permutation importance refinement
    p_imp = get_permutation_importance(bst, X_va_exp, y_val, prefilter,
                                       n_rounds=perm_rounds)
    refined = set(prefilter) | set(
        p_imp.loc[p_imp['importance_mean'] > 0, 'feature']
    )

    return sorted(refined)

if __name__ == '__main__':
    # --- Load UI parameters from S3 ---
    try:
        s3 = boto3.client('s3')
        bucket = 'ml-config'
        key    = 'run_config.json'
        obj    = s3.get_object(Bucket=bucket, Key=key)
        cfg    = json.load(obj['Body'])
    except Exception as e:
        raise RuntimeError(f"Error loading config from S3: {e}")

    # --- Read and merge datasets ---
    try:
        tables = {}
        for tbl in cfg['tables']:
            df = pd.read_csv(f"s3://{tbl['bucket']}/{tbl['key']}")
            tables[tbl['name']] = df[tbl['features']]
        df_main = None
        for j in cfg['joins']:
            left, right, on, how = j['left'], j['right'], j['on'], j.get('how', 'inner')
            if df_main is None:
                df_main = tables[left].merge(tables[right], on=on, how=how)
            else:
                df_main = df_main.merge(tables[right], on=on, how=how)
    except Exception as e:
        raise RuntimeError(f"Error reading or merging tables: {e}")

    # --- Split into train/test by feature ---
    try:
        split_cfg = cfg['split']
        col = split_cfg['column']
        val = split_cfg['value']
        if col not in df_main.columns:
            raise KeyError(f"Split column '{col}' not in data")
        train_mask = df_main[col] <= val
        df_train = df_main[train_mask]
        df_val   = df_main[~train_mask]
    except Exception as e:
        raise RuntimeError(f"Error in train/test split configuration: {e}")

    # --- Define features and labels ---
    try:
        X_train = df_train[cfg['features_train']]
        y_train = df_train[cfg['label_train']]
        X_val   = df_val[cfg['features_val']]
        y_val   = df_val[cfg['label_val']]
    except Exception as e:
        raise RuntimeError(f"Error selecting feature/label columns: {e}")

    # --- Load selection parameters ---
    prev_selected    = cfg.get('prev_selected', [])
    prev_near_misses = cfg.get('prev_near_misses', [])
    interactions     = cfg.get('interactions', [])
    K                = cfg.get('K', 100)
    perm_rounds      = cfg.get('perm_rounds', 10)

    # --- Run feature selection pipeline ---
    final_features = main(
        X_train, X_val, y_train, y_val,
        prev_selected, prev_near_misses,
        interactions, K, perm_rounds
    )

    # --- Output results ---
    print("Final feature list:")
    for feat in final_features:
        print(" -", feat)
