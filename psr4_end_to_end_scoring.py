"""
psr4_end_to_end_scoring.py
==========================================
Combined end-to-end pipeline for PSR-4 ACH+ Opt-In scoring.

Combines the preprocessing logic from feature_engineering_v3.py with the scoring
and top-drivers logic from optin_drivers_inference.py into a single script.

FLOW:
  RAW OOS FILE FROM S3
        │
        ▼
  STAGE 1: PREPROCESSING
    - Base filters (dedupe, PAYERSPONSORED=False, SUMCHECKPAYMENTCOUNT>0, DURATION>=90)
    - Null-fills (payments/phones → 0, TIME_SINCE_* → 9999)
    - 3 derived features (CHECK_PAYMENT_AVG_30, CHECK_PAYMENTCOUNT_MOMENTUM_30_90,
                          FIRSTACTION_BOOL)
    - One-hot encoding of PROVIDERTYPE only
    - Everything else kept as-is
        │
        ├──► Save intermediate to S3 (audit trail): oos_data_v3_<timestamp>.csv
        │
        │  (same DataFrame passed to Stage 2 in-memory)
        ▼
  STAGE 2: SCORING
    - Load champion XGBoost model from S3 (hardcoded)
    - Load train-derived cutoffs from local JSON
    - Compatibility check: verify model's features are present
    - Score with model.predict_proba()
    - Categorize into 5 tiers (Very High → Very Low)
    - Compute SHAP → top 5 drivers per row
    - Sort, rank, add CAMPAIGN_LEADS_RECOMMENDATION flag
        │
        ▼
  LOCAL FINAL CSV: optin_drivers_output_<timestamp>.csv
"""

import os
import io
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import boto3
import shap

from io import BytesIO, StringIO
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
BUCKET_NAME       = "zds-retraining-framework"

# Stage 1 — preprocessing I/O
INPUT_S3_KEY      = "Retraining_Framework_New/Input/New/OOS/raw_oos_data_20260608_124755.csv"
INTERMEDIATE_S3_FOLDER = "Retraining_Framework_New/Input/Preprocessed_Data/OOS/"
INTERMEDIATE_PREFIX    = "oos_data_v3"

# Stage 2 — scoring artifacts
MODEL_S3_KEY      = "Retraining_Framework_New/Output/classification_psr4_project__XGBClassifier__20260630_110203.pkl"

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
CUTOFFS_PATH      = os.path.join(SCRIPT_DIR, "train_cutoffs.json")
OUTPUT_DIR        = SCRIPT_DIR

# Column config
TRAIN_TARGET_COL   = 'OPT_IN_ACH'
OOS_TARGET_COL     = 'OPT_IN_ACH'
TRAIN_TIME_COL     = 'DURATION'
OOS_TIME_COL       = 'DURATION'
COMPOSITE_KEY_COLS = ['TIN', 'PROVIDERID', 'DATE_START', 'DATE_END']

TOP_N_DRIVERS   = 5
CATEGORY_LABELS = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# ==========================================
# FEATURE DESCRIPTIONS (for human-readable drivers)
# ==========================================
FEATURE_DESCRIPTIONS = {
    'SUMCHECKPAYMENTCOUNT90':
        'Total number of check payments in the last 90 days',
    'CHECK_PAYMENTCOUNT_MOMENTUM_30_90':
        'Ratio of daily check payment rate (last 30 days) to daily check payment rate (last 90 days)',
    'TIME_SINCE_LAST_CHECK_PAYMENT':
        'Days since the most recent check payment',
    'PROVIDERTYPE_MEDICAL':
        'Provider type is Medical',
    'CHECK_PAYMENT_AVG_30':
        'Average check payment amount in the last 30 days',
    'PHONECOUNT60':
        'Number of phone interactions in the last 60 days',
    'PAYERID_CHECK_COUNT':
        'Number of distinct payers issuing checks',
    'TOTAL_PREV_OPTOUTS_LAST_YEAR':
        'Number of prior ACH opt-outs in the last 12 months',
    'TOTAL_PREV_OPTOUTS':
        'Number of prior ACH opt-outs (lifetime)',
    'TIME_SINCE_LAST_CANCELLED_CHECK_PAYMENT':
        'Days since the most recent cancelled check payment',
    'FIRSTACTION_BOOL':
        'Whether this is a brand-new account (created on snapshot start)',
    'CANCELLEDCHECKPAYMENTCOUNT':
        'Total cancelled check payment count (lifetime)',
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def read_csv_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame:
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    return pd.read_csv(BytesIO(obj['Body'].read()))

def write_csv_to_s3(df, bucket, folder, filename_prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=f"{folder}{filename}", Body=csv_buffer.getvalue())
    return f"{folder}{filename}"

def load_pickle_from_s3(bucket_name: str, file_key: str):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    return pickle.load(io.BytesIO(obj['Body'].read()))

def reduce_mem_usage(df):
    """Iterate through all columns and modify data type to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'   -> Memory decreased by {100 * (start_mem - end_mem) / start_mem:.1f}% (Now: {end_mem:.1f} MB)')
    return df

def categorize_probability(prob: float, cutoffs: dict) -> str:
    """Assign a category based on train-derived cutoffs."""
    if prob >= cutoffs['Very_High']:
        return 'Very High'
    elif prob >= cutoffs['High']:
        return 'High'
    elif prob >= cutoffs['Medium']:
        return 'Medium'
    elif prob >= cutoffs['Low']:
        return 'Low'
    else:
        return 'Very Low'

def extract_top_drivers(shap_values: np.ndarray,
                        feature_values: pd.DataFrame,
                        feature_names: list,
                        top_n: int = 5) -> pd.DataFrame:
    """For each row, identify the top-N features by |SHAP value|, map to descriptions."""
    n_rows = shap_values.shape[0]
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap, axis=1)[:, -top_n:][:, ::-1]

    feature_names_arr  = np.array(feature_names)
    feature_values_arr = feature_values.values

    output = {}
    for rank in range(top_n):
        col_idx = top_indices[:, rank]
        raw_names = feature_names_arr[col_idx]
        descriptions = np.array([FEATURE_DESCRIPTIONS.get(name, name) for name in raw_names])
        output[f'top_driver_{rank+1}_name']  = descriptions
        output[f'top_driver_{rank+1}_value'] = feature_values_arr[np.arange(n_rows), col_idx]

    return pd.DataFrame(output)

# ==========================================
# =====                              ========
# =====   STAGE 1: PREPROCESSING     ========
# =====                              ========
# ==========================================
def run_base_preprocessing(df, time_col):
    """Apply base filters, null-fills, and compute the 3 derived features."""
    # --- Base filters ---
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[df['PAYERSPONSORED'] == False]

    fill_zero_cols = [
        'CANCELLEDCHECKPAYMENTCOUNT', 'CANCELLEDCHECKPAYMENTCOUNT30', 'CANCELLEDCHECKPAYMENTCOUNT60', 'CANCELLEDCHECKPAYMENTCOUNT90',
        'CANCELLEDCHECKPAYMENTAMOUNT', 'CANCELLEDCHECKPAYMENTAMOUNT30', 'CANCELLEDCHECKPAYMENTAMOUNT60', 'CANCELLEDCHECKPAYMENTAMOUNT90',
        'SUMCHECKPAYMENTAMOUNT', 'SUMCHECKPAYMENTCOUNT', 'SUMCHECKPAYMENTAMOUNT30', 'SUMCHECKPAYMENTCOUNT30',
        'SUMCHECKPAYMENTAMOUNT60', 'SUMCHECKPAYMENTCOUNT60', 'SUMCHECKPAYMENTAMOUNT90', 'SUMCHECKPAYMENTCOUNT90',
        'SUMACHPAYMENTAMOUNT', 'SUMACHPAYMENTCOUNT', 'SUMVCCPAYMENTAMOUNT', 'SUMVCCPAYMENTCOUNT',
        'SUMPAYERSPONSOREDPAYMENTAMOUNT', 'SUMPAYERSPONSOREDPAYMENTCOUNT', 'SUMTOTALPAYMENTAMOUNT', 'SUMTOTALPAYMENTCOUNT',
        'PREVIOUSOPTINCOUNT', 'PREVIOUSACHOPTINCOUNT', 'PREVIOUSVCCOPTINCOUNT',
        'PHONECOUNT', 'PHONECOUNT30', 'PHONECOUNT60', 'PHONECOUNT90'
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    if 'TIME_SINCE_LAST_CANCELLED_CHECK_PAYMENT' in df.columns:
        df['TIME_SINCE_LAST_CANCELLED_CHECK_PAYMENT'].fillna(9999, inplace=True)
    if 'TIME_SINCE_LAST_CHECK_PAYMENT' in df.columns:
        df['TIME_SINCE_LAST_CHECK_PAYMENT'].fillna(9999, inplace=True)

    df = df[df['SUMCHECKPAYMENTCOUNT'] > 0]
    if time_col in df.columns:
        df = df[df[time_col] >= 90]

    # --- Derived feature #1: CHECK_PAYMENT_AVG_30 ---
    df['CHECK_PAYMENT_AVG_30'] = df['SUMCHECKPAYMENTAMOUNT30'] / df['SUMCHECKPAYMENTCOUNT30'].replace(0, 1)
    df['CHECK_PAYMENT_AVG_30'].fillna(0, inplace=True)

    # --- Derived feature #2: CHECK_PAYMENTCOUNT_MOMENTUM_30_90 ---
    _check_rate_30 = df['SUMCHECKPAYMENTCOUNT30'] / 30
    _check_rate_90 = df['SUMCHECKPAYMENTCOUNT90'] / 90
    df['CHECK_PAYMENTCOUNT_MOMENTUM_30_90'] = _check_rate_30 / _check_rate_90.replace(0, 0.001)
    df['CHECK_PAYMENTCOUNT_MOMENTUM_30_90'].fillna(-1, inplace=True)

    # --- Derived feature #3: FIRSTACTION_BOOL ---
    if 'PROVIDERCREATEDON' in df.columns and 'DATE_START' in df.columns:
        df['FIRSTACTION_BOOL'] = (df['PROVIDERCREATEDON'] == df['DATE_START']).astype(int)

    df.reset_index(drop=True, inplace=True)

    # --- Categorical dtype (only PROVIDERTYPE, since it's the one being one-hot encoded) ---
    if 'PROVIDERTYPE' in df.columns: df['PROVIDERTYPE'] = df['PROVIDERTYPE'].astype('category')

    return df

def execute_encoding(df_raw):
    """One-hot encode PROVIDERTYPE only; everything else passes through as-is."""
    df_processed = df_raw.copy()
    all_features = list(df_processed.columns)
    print(f"   Carrying {len(all_features)} columns through encoding step.")

    df_in = df_processed[all_features].reset_index(drop=True)

    exclude_cols = ['TIN', OOS_TARGET_COL]
    cat_cols = ['PROVIDERTYPE'] if 'PROVIDERTYPE' in df_in.columns else []
    passthrough_cols = [c for c in df_in.columns
                        if c not in exclude_cols
                        and c not in cat_cols]

    print(f"   Encoding: {cat_cols}")
    print(f"   Passthrough ({len(passthrough_cols)} cols): kept as-is")

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    if cat_cols:
        encoded_cats = encoder.fit_transform(df_in[cat_cols])
        df_cat = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))
    else:
        df_cat = pd.DataFrame()

    df_passthrough = df_in[passthrough_cols].reset_index(drop=True)

    df_final = pd.concat([df_in[['TIN', OOS_TARGET_COL]].reset_index(drop=True),
                          df_passthrough,
                          df_cat], axis=1)
    df_final = reduce_mem_usage(df_final)
    return df_final

# ==========================================
# =====                              ========
# =====   STAGE 2: SCORING           ========
# =====                              ========
# ==========================================
def score_and_enrich(df_encoded, model, model_features, cutoffs):
    """
    Take the preprocessed DataFrame and produce the final scored output.
    Returns the enriched DataFrame with probability, category, drivers, ranks, and campaign flag.
    """
    # ----- Compatibility check -----
    missing = [f for f in model_features if f not in df_encoded.columns]
    if missing:
        raise ValueError(
            f"Preprocessed output is missing {len(missing)} model feature(s): {missing}"
        )

    # ----- Composite key check -----
    missing_key_cols = [c for c in COMPOSITE_KEY_COLS if c not in df_encoded.columns]
    if missing_key_cols:
        raise ValueError(f"Composite key columns missing: {missing_key_cols}")

    # ----- Build model input -----
    X_oos = df_encoded[model_features].copy()
    print(f"   Feature matrix shape: {X_oos.shape}")

    # ----- Score -----
    print("   Scoring with model.predict_proba()...")
    probabilities = model.predict_proba(X_oos)[:, 1]
    print(f"   Scored {len(probabilities):,} rows. "
          f"Prob range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")

    # ----- Categorize -----
    print("   Categorizing probabilities...")
    categories = np.array([categorize_probability(p, cutoffs) for p in probabilities])
    dist = pd.Series(categories).value_counts(normalize=True).reindex(CATEGORY_LABELS[::-1]).fillna(0)
    print("   Category distribution:")
    for cat, pct in dist.items():
        print(f"     {cat:<12} {pct*100:5.2f}%")

    # ----- SHAP drivers -----
    print("   Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_oos)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(f"   SHAP matrix shape: {shap_values.shape}")
    print(f"   Extracting top-{TOP_N_DRIVERS} drivers per row...")
    drivers_df = extract_top_drivers(shap_values, X_oos, model_features, top_n=TOP_N_DRIVERS)

    # ----- Assemble output -----
    df_encoded = df_encoded.reset_index(drop=True)
    drivers_df = drivers_df.reset_index(drop=True)

    df_out = df_encoded.copy()
    df_out['OPT_IN_PROBABILITY'] = probabilities
    df_out['OPT_IN_CATEGORY']    = categories
    df_out = pd.concat([df_out, drivers_df], axis=1)
    df_out['CAMPAIGN_LEADS_RECOMMENDATION'] = (df_out['OPT_IN_CATEGORY'] == 'Very High').astype(int)

    # ----- Sort: category (Very High → Very Low), then probability desc -----
    category_order = pd.CategoricalDtype(
        categories=['Very High', 'High', 'Medium', 'Low', 'Very Low'],
        ordered=True
    )
    df_out['OPT_IN_CATEGORY'] = df_out['OPT_IN_CATEGORY'].astype(category_order)

    df_out = df_out.sort_values(
        by=['OPT_IN_CATEGORY', 'OPT_IN_PROBABILITY'],
        ascending=[True, False]
    ).reset_index(drop=True)

    df_out['OPT_IN_ORDER_WITHIN_CATEGORY'] = (
        df_out.groupby('OPT_IN_CATEGORY').cumcount() + 1
    )
    df_out['CALL_TARGET_ORDER'] = np.arange(1, len(df_out) + 1)

    df_out['OPT_IN_CATEGORY'] = df_out['OPT_IN_CATEGORY'].astype(str)

    # Reorder new columns: probability, category, ranks, drivers, campaign flag
    driver_cols = [c for c in df_out.columns if c.startswith('top_driver_')]
    new_cols_ordered = (
        ['OPT_IN_PROBABILITY', 'OPT_IN_CATEGORY',
         'OPT_IN_ORDER_WITHIN_CATEGORY', 'CALL_TARGET_ORDER']
        + driver_cols
        + ['CAMPAIGN_LEADS_RECOMMENDATION']
    )
    other_cols = [c for c in df_out.columns if c not in new_cols_ordered]
    df_out = df_out[other_cols + new_cols_ordered]

    return df_out

# ==========================================
# MAIN ORCHESTRATION
# ==========================================
def main():
    print("=" * 70)
    print("PSR-4 End-to-End Scoring Pipeline")
    print("=" * 70)

    # ==========================================
    # STAGE 1 — PREPROCESSING (independent try block)
    # ==========================================
    try:
        print(f"\n[STAGE 1] Preprocessing")
        print(f"  Fetching input from S3: s3://{BUCKET_NAME}/{INPUT_S3_KEY}")
        df_raw = read_csv_from_s3(BUCKET_NAME, INPUT_S3_KEY)
        print(f"  Input shape: {df_raw.shape}")

        print("\n  Running base preprocessing...")
        df_processed = run_base_preprocessing(df_raw, OOS_TIME_COL)
        print(f"  After preprocessing shape: {df_processed.shape}")

        print("\n  Running encoding step...")
        df_encoded = execute_encoding(df_processed)
        print(f"  After encoding shape: {df_encoded.shape}")

        print("\n  Saving intermediate audit artifact to S3...")
        intermediate_key = write_csv_to_s3(df_encoded, BUCKET_NAME,
                                           INTERMEDIATE_S3_FOLDER, INTERMEDIATE_PREFIX)
        print(f"  ✅ Intermediate saved: s3://{BUCKET_NAME}/{intermediate_key}")

    except Exception as e:
        print(f"\n❌ STAGE 1 FAILED: {e}")
        return False

    # ==========================================
    # STAGE 2 — SCORING (independent try block)
    # ==========================================
    try:
        print(f"\n[STAGE 2] Scoring")

        print(f"  Loading champion model from S3: {MODEL_S3_KEY}")
        loaded_obj = load_pickle_from_s3(BUCKET_NAME, MODEL_S3_KEY)
        if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
            model = loaded_obj['model']
            model_features = loaded_obj.get('features')
        else:
            model = loaded_obj
            model_features = getattr(model, 'feature_names_in_', None)

        if model_features is None:
            raise ValueError("Could not extract feature list from the model pickle.")
        model_features = list(model_features)
        print(f"  Model expects {len(model_features)} features.")

        print(f"\n  Loading cutoffs from: {CUTOFFS_PATH}")
        with open(CUTOFFS_PATH, 'r') as f:
            cutoffs_json = json.load(f)
        cutoffs = cutoffs_json.get('cutoffs', cutoffs_json)
        print(f"  Cutoffs: Very_High={cutoffs['Very_High']:.4f}, "
              f"High={cutoffs['High']:.4f}, Medium={cutoffs['Medium']:.4f}, "
              f"Low={cutoffs['Low']:.4f}")

        print("\n  Scoring and enriching...")
        df_out = score_and_enrich(df_encoded, model, model_features, cutoffs)
        print(f"  Final output shape: {df_out.shape}")
        print(f"  Very High rows (CAMPAIGN_LEADS = 1): "
              f"{(df_out['CAMPAIGN_LEADS_RECOMMENDATION'] == 1).sum():,}")

        # ----- Save locally -----
        timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"optin_drivers_output_{timestamp}.csv"
        output_path     = os.path.join(OUTPUT_DIR, output_filename)
        df_out.to_csv(output_path, index=False)
        print(f"\n✅ Final output saved locally: {output_path}")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ STAGE 2 FAILED: {e}")
        print(f"   Note: Stage 1's audit artifact was saved to S3 successfully.")
        print(f"   You can re-run Stage 2 alone using that file.")
        return False


if __name__ == "__main__":
    main()
