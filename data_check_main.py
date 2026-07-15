import pandas as pd

import numpy as np

from datetime import datetime

import warnings

import boto3

from io import BytesIO
 
warnings.filterwarnings('ignore')

from pdf_reporter import DataValidationReporter

from catalog_loader import load_model_catalog, get_s3_config
 
# --- 1. Hardcoded Configuration ---

DUPLICATE_SUBSET_COLS =  ['TIN','PROVIDERID', 'DATE_START','DATE_END']
 
ALT_MODEL2_CRITICAL_FEATURES = [

    'PROVIDERTYPE_MEDICAL', 'SUMCHECKPAYMENTCOUNT30', 'SUMCHECKPAYMENTAMOUNT30',

    'CHECK_PAYMENT_AVG_30', 'CHECK_PAYMENTCOUNT_MOMENTUM_30_90', 'PAYERID_CHECK_COUNT',

    'TIME_SINCE_LAST_CHECK_PAYMENT', 'CANCELLEDCHECKPAYMENTCOUNT30',

    'TIME_SINCE_LAST_CANCELLED_CHECK_PAYMENT', 'CHECK_CONTAINS_METLIFE',

    'FIRSTACTION_BOOL', 'PREVIOUSACHOPTINCOUNT'

]
 
NOT_CONSIDERED_COLUMNS = [


]
 
# Dates (informational metadata shown in the PDF report)

train_date_from = "2024-10-01"

train_date_to = "2025-09-30"

val_date_from = "2025-10-01"

val_date_to = "2025-12-31"

oos_date_from = "2026-01-01"

oos_date_to = ""
 
# S3 Functions

def get_latest_file_from_s3(bucket_name: str, folder_prefix: str) -> str:

    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")

    pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)

    latest = None

    for page in pages:

        for obj in page.get("Contents", []):

            if obj["Key"].endswith("/"): continue

            if latest is None or obj["LastModified"] > latest["LastModified"]:

                latest = obj

    return latest["Key"]
 
def read_csv_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame:

    s3 = boto3.client("s3")

    obj = s3.get_object(Bucket=bucket_name, Key=file_key)

    return pd.read_csv(BytesIO(obj['Body'].read()))
 
print('--- Data Loading ---')

_catalog = load_model_catalog()
bucket_name, _folders = get_s3_config(_catalog)
train_prefix = _folders["raw_train"]
val_prefix   = _folders["raw_val"]
oos_prefix   = _folders["raw_oos"]

train_key = get_latest_file_from_s3(bucket_name, train_prefix)

val_key   = get_latest_file_from_s3(bucket_name, val_prefix)

oos_key   = get_latest_file_from_s3(bucket_name, oos_prefix)

train_df = read_csv_from_s3(bucket_name, train_key)

val_df   = read_csv_from_s3(bucket_name, val_key)

oos_df   = read_csv_from_s3(bucket_name, oos_key)

print("Downloaded Train, Val & OOS CSVs ✅")

# --- 1b. EDA: Pandas Profiling on Combined Dataset ---

print("\n--- Running EDA Profiling on Combined Dataset ---")

try:

    from ydata_profiling import ProfileReport

    # Combine all 3 datasets so the profile reflects the full data picture.
    # Add a 'DATA_SPLIT' column so the profile can show distribution per split.

    train_df_eda = train_df.copy()
    train_df_eda['DATA_SPLIT'] = 'Train'

    val_df_eda = val_df.copy()
    val_df_eda['DATA_SPLIT'] = 'Validation'

    oos_df_eda = oos_df.copy()
    oos_df_eda['DATA_SPLIT'] = 'OOS'

    combined_df = pd.concat([train_df_eda, val_df_eda, oos_df_eda], axis=0, ignore_index=True)

    print(f"Combined dataset shape: {combined_df.shape[0]:,} rows x {combined_df.shape[1]} cols")

    # minimal=True skips slow correlations/interactions for tractable runtime on large data.
    profile = ProfileReport(
        combined_df,
        title="EDA Profile: Combined Train + Validation + OOS",
        minimal=True,
        explorative=False,
    )

    eda_filename = f"EDA_Profile_Combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    profile.to_file(eda_filename)

    print(f"✅ EDA profile saved locally as: {eda_filename}")

    # Free the combined frame to keep memory usage in check before later steps.
    del combined_df, train_df_eda, val_df_eda, oos_df_eda, profile

except ImportError:

    print("⚠️ ydata-profiling not installed. Skipping EDA. Run: pip install ydata-profiling")

except Exception as e:

    print(f"⚠️ EDA profiling failed: {e}. Continuing with data validation tests.")

# --- 2. DYNAMIC COLUMN ROUTING ---

print("--- Auto-Routing Feature Columns ---")

# 1. Grab all numericals

numeric_cols_outlier = train_df.select_dtypes(include=np.number).columns.tolist()
 
# 2. Grab all categoricals/strings

cat_cols_raw = train_df.select_dtypes(exclude=np.number).columns.tolist()
 
cat_cols_fixed = []

long_tail_cols = []

text_cols_length_distribution = []
 
# 3. Route categoricals based on Cardinality threshold (50)

for col in cat_cols_raw:

    if col in NOT_CONSIDERED_COLUMNS:

        continue

    unique_count = train_df[col].nunique()

    if unique_count < 50:

        cat_cols_fixed.append(col)

    else:

        long_tail_cols.append(col)

        text_cols_length_distribution.append(col)
 
print(f"Routed {len(numeric_cols_outlier)} Numeric Cols")

print(f"Routed {len(cat_cols_fixed)} Low-Cardinality Categoricals")

print(f"Routed {len(long_tail_cols)} High-Cardinality Strings/IDs")
 
# --- 3. Main Execution ---

if __name__ == "__main__":

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ----------------------------------------------------------
    # Comparison 1: Train vs Validation
    # ----------------------------------------------------------

    print("\n--- Starting Data Validation Orchestration: Train vs Validation ---")

    reporter_val = DataValidationReporter(

        train_df,

        val_df,

        not_considered_columns=NOT_CONSIDERED_COLUMNS,

        cat_cols_fixed=cat_cols_fixed,

        numeric_cols_outlier=numeric_cols_outlier,

        long_tail_cols=long_tail_cols,

        text_cols_length_distribution=text_cols_length_distribution,

        duplicate_subset_cols=DUPLICATE_SUBSET_COLS,

        alt_model_features=ALT_MODEL2_CRITICAL_FEATURES,

        train_date_from=train_date_from,

        train_date_to=train_date_to,

        oos_date_from=val_date_from,

        oos_date_to=val_date_to,

    )

    reporter_val.run_all_tests()

    val_report_filename = f"Data_Consistency_Report_TRAIN_vs_VAL_{timestamp}.pdf"

    reporter_val.generate_report(filename=val_report_filename)

    print(f"Train vs Val report saved as: {val_report_filename}")

    # ----------------------------------------------------------
    # Comparison 2: Train vs OOS
    # ----------------------------------------------------------

    print("\n--- Starting Data Validation Orchestration: Train vs OOS ---")

    reporter_oos = DataValidationReporter(

        train_df,

        oos_df,

        not_considered_columns=NOT_CONSIDERED_COLUMNS,

        cat_cols_fixed=cat_cols_fixed,

        numeric_cols_outlier=numeric_cols_outlier,

        long_tail_cols=long_tail_cols,

        text_cols_length_distribution=text_cols_length_distribution,

        duplicate_subset_cols=DUPLICATE_SUBSET_COLS,

        alt_model_features=ALT_MODEL2_CRITICAL_FEATURES,

        train_date_from=train_date_from,

        train_date_to=train_date_to,

        oos_date_from=oos_date_from,

        oos_date_to=oos_date_to,

    )

    reporter_oos.run_all_tests()

    oos_report_filename = f"Data_Consistency_Report_TRAIN_vs_OOS_{timestamp}.pdf"

    reporter_oos.generate_report(filename=oos_report_filename)

    print(f"Train vs OOS report saved as: {oos_report_filename}")

    print(f"\n--- Orchestration Complete ---")
 


 📋 Horizontal list for feature_engineering.py:
# ['PAYERID_CHECK_COUNT', 'SUMCHECKPAYMENTCOUNT90', 'CANCELLEDCHECKPAYMENTCOUNT', 'PHONECOUNT60', 'PROVIDERTYPE_MEDICAL', 'TIME_SINCE_LAST_CHECK_PAYMENT', 'TIME_SINCE_LAST_CANCELLED_CHECK_PAYMENT', 'TOTAL_PREV_OPTOUTS', 'TOTAL_PREV_OPTOUTS_LAST_YEAR', 'SEGMENT_Petite', 'SUMACHPAYMENTAMOUNT', 'SUMPAYERSPONSOREDPAYMENTCOUNT', 'APV', 'DONOTCALL', 'GLOBALEXCLUDED']