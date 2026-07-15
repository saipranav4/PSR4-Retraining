import pandas as pd

import boto3

from datetime import datetime

from io import StringIO

import snowflake.connector

from catalog_loader import load_model_catalog, get_s3_config

# =========================================================

# 1. GLOBAL CONFIGURATION

# =========================================================

# Hard date cutoffs (three-bucket split):
#   Train:      DATE_END <  TRAIN_END_DATE
#   Validation: TRAIN_END_DATE <= DATE_END < VAL_END_DATE
#   OOS:        DATE_END >= VAL_END_DATE

TRAIN_END_DATE = '2025-10-01'
VAL_END_DATE   = '2026-01-01'

# Change this if your database uses a different column for chronological sorting

DATE_COLUMN = 'DATE_END'

# =========================================================

# 2. FUNCTION: Upload DataFrame to S3

# =========================================================

def upload_df_to_s3(df, bucket_name, s3_folder, file_prefix):

    """Converts a DataFrame to CSV in memory and uploads it to S3."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{file_prefix}_{timestamp}.csv"

    csv_buffer = StringIO()

    df.to_csv(csv_buffer, index=False)

    s3 = boto3.client("s3")

    s3.put_object(

        Bucket=bucket_name,

        Key=f"{s3_folder}{filename}",

        Body=csv_buffer.getvalue()

    )

    print(f"✅ Uploaded: s3://{bucket_name}/{s3_folder}{filename} ({df.shape[0]:,} rows)")

# =========================================================

# 3. FUNCTION: Fetch Master Data

# =========================================================

def fetch_master_data(query, snowflake_config):

    """Executes a Snowflake query and returns a pandas DataFrame."""

    print("🔌 Connecting to Snowflake...")

    conn = snowflake.connector.connect(**snowflake_config)

    try:

        print("⏳ Executing master query...")

        df = pd.read_sql(query, conn)

        df.columns = [col.upper() for col in df.columns]

        print(f"✅ Master data downloaded successfully: {df.shape[0]:,} total rows")

        return df

    finally:

        conn.close()

# =========================================================

# 4. MAIN PIPELINE

# =========================================================

def main():

    print(f"🚀 Starting Date-Cutoff Split Pipeline:")
    print(f"   Train: {DATE_COLUMN} <  {TRAIN_END_DATE}")
    print(f"   Val:   {TRAIN_END_DATE} <= {DATE_COLUMN} < {VAL_END_DATE}")
    print(f"   OOS:   {DATE_COLUMN} >= {VAL_END_DATE}")

    print(f"📅 Using '{DATE_COLUMN}' as the split anchor.\n")

    # -------------------------------

    # Snowflake Configuration

    # -------------------------------

    snowflake_config = {
        "user": 'SVC_KNIME_DEV',

        "password": '1bhYzI$Nq1evxo=',

        "account": 'ti97672.east-us-2.azure',

        "database": ' WORKSPACE_DS_OFFSHORE',

        "schema": 'MODEL',

        "warehouse": 'DEV_DATA_ANALYST'

    }

    # -------------------------------

    # S3 Configuration

    # -------------------------------

    _catalog = load_model_catalog()
    bucket_name, _folders = get_s3_config(_catalog)
    train_folder = _folders["raw_train"]
    val_folder   = _folders["raw_val"]
    oos_folder   = _folders["raw_oos"]

    # -------------------------------

    # Master Query

    # -------------------------------

    master_query = """

     SELECT * FROM WORKSPACE_DS_OFFSHORE.MODEL.PSR4_PSR5_RF_TRAINING;

    """

    try:

        # 1. Fetch All Data

        master_df = fetch_master_data(master_query, snowflake_config)

        # Ensure the target date column exists to prevent key errors

        if DATE_COLUMN not in master_df.columns:

            raise KeyError(f"Column '{DATE_COLUMN}' not found in Snowflake data. Available columns: {master_df.columns.tolist()}")

        # Convert to datetime object

        master_df[DATE_COLUMN] = pd.to_datetime(master_df[DATE_COLUMN])

        # Step A: Sort the entire dataframe chronologically

        print(f"⏳ Sorting data chronologically by {DATE_COLUMN}...")

        master_df = master_df.sort_values(by=DATE_COLUMN).reset_index(drop=True)

        # -------------------------------

        # 2. Apply Hard Date Cutoff

        # -------------------------------

        total_rows = len(master_df)

        train_end_date = pd.to_datetime(TRAIN_END_DATE)
        val_end_date   = pd.to_datetime(VAL_END_DATE)

        print(f"--- Hard Date Cutoffs ---")

        print(f"Train ends (Val starts): {train_end_date.date()}")
        print(f"Val ends (OOS starts):   {val_end_date.date()}")

        # -------------------------------

        # 3. Perform Strict Date-Based Slicing (3 buckets)

        # -------------------------------

        train_df = master_df[master_df[DATE_COLUMN] < train_end_date].copy()

        val_df   = master_df[(master_df[DATE_COLUMN] >= train_end_date) &
                             (master_df[DATE_COLUMN] < val_end_date)].copy()

        oos_df   = master_df[master_df[DATE_COLUMN] >= val_end_date].copy()

        # -------------------------------

        # 5. VERIFICATION: Check Date Ranges

        # -------------------------------

        train_max = train_df[DATE_COLUMN].max() if len(train_df) > 0 else None
        val_min   = val_df[DATE_COLUMN].min()   if len(val_df) > 0 else None
        val_max   = val_df[DATE_COLUMN].max()   if len(val_df) > 0 else None
        oos_min   = oos_df[DATE_COLUMN].min()   if len(oos_df) > 0 else None
        oos_max   = oos_df[DATE_COLUMN].max()   if len(oos_df) > 0 else None

        print(f"\n--- Date Range Verification ---")

        if train_max is not None:
            print(f"Train: {train_df[DATE_COLUMN].min().date()}  ->  {train_max.date()}")
        if val_min is not None:
            print(f"Val:   {val_min.date()}  ->  {val_max.date()}")
        if oos_min is not None:
            print(f"OOS:   {oos_min.date()}  ->  {oos_max.date()}")

        # Check if dates are collapsing/overlapping

        boundaries_ok = True
        if train_max is not None and val_min is not None and train_max >= val_min:
            print(f"\n❌ CRITICAL WARNING: Train and Val boundaries overlap! Train Max ({train_max}) >= Val Min ({val_min})")
            boundaries_ok = False
        if val_max is not None and oos_min is not None and val_max >= oos_min:
            print(f"\n❌ CRITICAL WARNING: Val and OOS boundaries overlap! Val Max ({val_max}) >= OOS Min ({oos_min})")
            boundaries_ok = False

        if boundaries_ok:
            print("✅ Verification Passed: No date ranges are collapsing or overlapping!")

        # -------------------------------

        # 6. Print Final Dataset Sizes

        # -------------------------------

        print(f"\n🎯 Final Dataset Sizes:")

        print(f"   -> Training Set:   {train_df.shape[0]:,} rows ({len(train_df)/total_rows*100:.1f}%)")

        print(f"   -> Validation Set: {val_df.shape[0]:,} rows ({len(val_df)/total_rows*100:.1f}%)")

        print(f"   -> OOS Set:        {oos_df.shape[0]:,} rows ({len(oos_df)/total_rows*100:.1f}%)")

        # -------------------------------

        # 7. Upload to S3

        # -------------------------------

        print("\n--- Uploading Datasets to S3 ---")

        upload_df_to_s3(train_df, bucket_name, train_folder, "raw_train_data")

        upload_df_to_s3(val_df, bucket_name, val_folder, "raw_val_data")

        upload_df_to_s3(oos_df, bucket_name, oos_folder, "raw_oos_data")

        print("\n🎉 Pipeline execution finished successfully!")

    except Exception as e:

        print(f"\n❌ Error in pipeline: {e}")

if __name__ == "__main__":

    main()
 