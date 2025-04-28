import streamlit as st
import boto3
import pandas as pd
import json

# AWS S3 config
S3_BUCKET = st.sidebar.text_input("S3 Bucket for data/CSV files", "my-data-bucket")
S3_PREFIX = st.sidebar.text_input("S3 Prefix for CSV files", "datasets/")
CONFIG_BUCKET = st.sidebar.text_input("Config Bucket for saving JSON", "ml-config")
CONFIG_KEY = st.sidebar.text_input("Config Key (JSON file)", "run_config.json")

@st.cache_data(ttl=600)
def list_csv_files(bucket, prefix):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    files = []
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.lower().endswith('.csv'):
                files.append(key)
    return files

@st.cache_data(ttl=600)
def load_csv_head(bucket, key, nrows=5):
    url = f"s3://{bucket}/{key}"
    df = pd.read_csv(url, nrows=nrows)
    return list(df.columns)

def main():
    st.title("Feature Engineering Config UI")

    st.markdown("## 1. Select source CSV tables and features")
    csv_files = list_csv_files(S3_BUCKET, S3_PREFIX)
    selected_tables = st.multiselect("Choose CSV files", csv_files)

    table_configs = []
    for tbl_key in selected_tables:
        st.markdown(f"### Table: {tbl_key}")
        cols = load_csv_head(S3_BUCKET, tbl_key)
        features = st.multiselect(f"Features from {tbl_key}", cols, key=f"feat_{tbl_key}")
        table_configs.append({
            'name': tbl_key.replace('/', '_').replace('.csv',''),
            'bucket': S3_BUCKET,
            'key': tbl_key,
            'features': features
        })

    st.markdown("## 2. Define joins")
    joins = []
    if len(table_configs) >= 2:
        num_joins = st.number_input("Number of joins", min_value=0, max_value=len(table_configs)-1, value=1, step=1)
        for i in range(int(num_joins)):
            left = st.selectbox(f"Join {i+1} - Left table", [t['name'] for t in table_configs], key=f"join_left_{i}")
            right = st.selectbox(f"Join {i+1} - Right table", [t['name'] for t in table_configs if t['name']!=left], key=f"join_right_{i}")
            on = st.text_input(f"Join {i+1} - On column", key=f"join_on_{i}")
            how = st.selectbox(f"Join {i+1} - How", ['inner','left','right','outer'], key=f"join_how_{i}")
            joins.append({'left': left, 'right': right, 'on': on, 'how': how})

    st.markdown("## 3. Split data into train/validation")
    split_column = st.text_input("Split column name (e.g. Year)")
    split_value = st.text_input("Split threshold value (e.g. 2020)")

    st.markdown("## 4. Label and feature lists")
    features = st.multiselect("Final features for both train and validation", [], key="final_features", help="Select after table configs loaded")
    label = st.text_input("Label column name")

    st.markdown("## 5. Define interactions")
    ops = ['ratio','diff','prod','sum']
    interactions = []
    num_int = st.number_input("Number of interactions", min_value=0, value=0, step=1)
    for i in range(int(num_int)):
        new_name = st.text_input(f"Interaction {i+1} name", key=f"int_name_{i}")
        fi = st.selectbox(f"Interaction {i+1} - Feature i", features, key=f"int_fi_{i}")
        fj = st.selectbox(f"Interaction {i+1} - Feature j", features, key=f"int_fj_{i}")
        op_key = st.selectbox(f"Interaction {i+1} - Operation", ops, key=f"int_op_{i}")
        interactions.append([new_name, fi, fj, op_key])

    st.markdown("## 6. Define lag features")
    lag_specs = []
    num_lags = st.number_input("Number of lag specs", min_value=0, value=0, step=1)
    for i in range(int(num_lags)):
        feat = st.selectbox(f"Lag {i+1} - Feature", features, key=f"lag_feat_{i}")
        lag = st.number_input(f"Lag {i+1} - periods", min_value=1, value=1, step=1, key=f"lag_period_{i}")
        lag_specs.append([feat, lag])

    st.markdown("## 7. Other parameters")
    prev_selected = st.multiselect("Always-keep features", features, [])
    prev_near_misses = st.multiselect("Near-miss features", features, [])
    K = st.number_input("Top-K for pre-filtering", min_value=1, value=100, step=1)
    perm_rounds = st.number_input("Permutation rounds", min_value=1, value=10, step=1)

    st.markdown("## 8. Final dataset name & Save config")
    final_dataset = st.text_input("Final dataset name (prefix for S3)", value="feature_config")

    if st.button("Save configuration to S3"):
        config = {
            'tables': table_configs,
            'joins': joins,
            'split': {'column': split_column, 'value': split_value},
            'features': features,
            'label': label,
            'interactions': interactions,
            'lag_specs': lag_specs,
            'prev_selected': prev_selected,
            'prev_near_misses': prev_near_misses,
            'K': K,
            'perm_rounds': perm_rounds
        }
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket=CONFIG_BUCKET,
            Key=f"{final_dataset}.json",
            Body=json.dumps(config)
        )
        st.success(f"Configuration saved to s3://{CONFIG_BUCKET}/{final_dataset}.json")

if __name__ == '__main__':
    main()
