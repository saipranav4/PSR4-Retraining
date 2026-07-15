import pandas as pd

import numpy as np

import boto3

import warnings

from io import BytesIO, StringIO

from datetime import datetime

from itertools import combinations

from catalog_loader import load_model_catalog, get_s3_config
 
# Sklearn Classification Imports

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

from scipy.stats import chi2_contingency
 
warnings.filterwarnings('ignore')
 
# ==========================================

# 0. GLOBAL CONFIGURATION

# ==========================================

TRAIN_TARGET_COL = 'OPT_IN_ACH'

OOS_TARGET_COL = 'OPT_IN_ACH'

TRAIN_TIME_COL = 'DURATION'

OOS_TIME_COL = 'DURATION'
 
# ==========================================

# 1. AWS S3 & MEMORY HELPER FUNCTIONS

# ==========================================
 
def get_latest_file_from_s3(bucket_name: str, folder_prefix: str) -> str:

    s3 = boto3.client("s3")

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)    

    if 'Contents' not in response:

        raise ValueError(f"No files found in {folder_prefix}")    

    files = [obj for obj in response['Contents'] if not obj['Key'].endswith('/')]    

    if not files:

        raise ValueError(f"No files found in {folder_prefix}")    

    latest_file = max(files, key=lambda x: x['LastModified'])

    return latest_file['Key']
 
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
 
# ==========================================

# 2. FEATURE ENGINEERING MATH

# ==========================================
 #derived new
def PHONE_CALL_DENSITY(df, windows):

    for window in windows:

        phone_col = f'PHONECOUNT{window}'

        cancelled_col = f'CANCELLEDCHECKPAYMENTCOUNT{window}'

        density_col = f'PHONE_CALL_DENSITY_{window}'

        if phone_col in df.columns and cancelled_col in df.columns:

            df[density_col] = (df[phone_col] / df[cancelled_col].replace(0, pd.NA)).fillna(0)

    return df
 #derived new
def Cancelled_Check_Amount_Ratio(df, windows):

    for window in windows:

        cancelled_amount_col = f'CANCELLEDCHECKPAYMENTAMOUNT{window}'

        total_amount_col = f'SUMCHECKPAYMENTAMOUNT{window}'

        ratio_col = f'CANCELLED_CHECK_AMOUNT_RATIO{window}'

        if cancelled_amount_col in df.columns and total_amount_col in df.columns:

            df[ratio_col] = (df[cancelled_amount_col] / df[total_amount_col].replace(0, pd.NA)).fillna(0)

    return df
 #derived new
def CHECK_PAYMENT_STABILITY(df, windows):

    windows = sorted(windows) 

    for x, y in combinations(windows, 2):

        col_x = f'CHECK_PAYMENT_AVG_{x}'

        col_y = f'CHECK_PAYMENT_AVG_{y}'

        stability_col = f'CHECK_PAYMENT_STABILITY_{x}_{y}'

        if col_x in df.columns and col_y in df.columns:

            df[stability_col] = (df[col_x] / df[col_y].replace(0, pd.NA)).fillna(1.0)

            df[stability_col] = df[stability_col].replace([float('inf'), float('-inf')], 9999)

    return df
 
def LONG_STARTER_BOOL(df, days, time_col):

    col_x = 'FIRSTACTION_BOOL'

    if col_x in df.columns and time_col in df.columns:

        df['LONG_UNDECIDED_BOOL'] = ((df[col_x] == 1) & (df[time_col] > days)).astype(int)

    return df
 
def independent_features(df, time_col): #all are newly derived

    if 'PREVIOUSACHOPTINCOUNT' in df.columns and 'PREVIOUSOPTINCOUNT' in df.columns:

        df['ACH_PREFERENCE_RATIO'] = (df['PREVIOUSACHOPTINCOUNT'].fillna(0) / df['PREVIOUSOPTINCOUNT'].replace(0, 1))
    #derived new
    df['Avg_Days_Between_Check_Payments'] = (df[time_col] / df['SUMCHECKPAYMENTCOUNT'].replace(0, 1))
    #derived new
    df['ENGAGEMENT_ADJUSTED_FRICTION'] = (df['TIME_SINCE_LAST_CANCELLED_CHECK_PAYMENT'] / (df['PHONECOUNT'] + 1))
    #derived new
    df['STATUS_RECENCY_GAP'] = (df['TIME_SINCE_LAST_CHECK_PAYMENT'] - df['TIME_SINCE_LAST_CANCELLED_CHECK_PAYMENT'])

    # df['PAIN_POINT_RATIO'] = np.log1p((df['PHONECOUNT90'] + 1) / (df['SUMCHECKPAYMENTCOUNT90'] + 1))
    #derived new
    df['REVENUE_VELOCITY'] = (df['SUMCHECKPAYMENTAMOUNT30'] / (df['SUMCHECKPAYMENTAMOUNT90'] / 3).replace(0, 1)) - 1

    # df['PAPER_FATIGUE'] = (df['SUMCHECKPAYMENTCOUNT90'] / df['PAYERID_CHECK_COUNT'].replace(0, 1))

    avg_days_between_checks = (df[time_col] / df['SUMCHECKPAYMENTCOUNT'].replace(0, 1))

    # df['RECENCY_SEVERITY'] = (df['TIME_SINCE_LAST_CHECK_PAYMENT'] / avg_days_between_checks.replace(0, 1))

    # df['CANCEL_SHOCK'] = (df['CANCELLEDCHECKPAYMENTCOUNT30'] > (df['CANCELLEDCHECKPAYMENTCOUNT90'] / 3)).astype(int)

    df['UNDECIDED_HIGH_ROLLER'] = (df['FIRSTACTION_BOOL'] * df['SUMCHECKPAYMENTAMOUNT90'])

    df['CHECK_DENSITY'] = (df['SUMCHECKPAYMENTCOUNT'] / df[time_col].replace(0, 1))

    # df['ACTIVE_FRUSTRATION'] = (df['PHONECOUNT30'] / df['TIME_SINCE_LAST_CHECK_PAYMENT'].replace(0, 1))

    df['PAYMENT_ACCELERATION'] = ((df['CHECK_PAYMENTCOUNT_RATE_30'] - df['CHECK_PAYMENTCOUNT_RATE_90']) / df['CHECK_PAYMENTCOUNT_RATE_90'].replace(0, 0.001))

    df['MOMENTUM_30_LIFETIME'] = (df['CHECK_PAYMENTCOUNT_RATE_30'] / df['CHECK_PAYMENTCOUNT_RATE'].replace(0, 0.001))

    df['PAYMENT_VOLATILITY_INDEX'] = ((df['CHECK_PAYMENT_AVG_30'] - df['CHECK_PAYMENT_AVG_90']).abs() / df['CHECK_PAYMENT_AVG_90'].replace(0, 1))

    df['CHECK_FAILURE_RATE_90'] = (df['CANCELLEDCHECKPAYMENTCOUNT90'] / df['SUMCHECKPAYMENTCOUNT90'].replace(0, 1))

    epsilon = 1e-6

    den = np.log1p(df['SUMCHECKPAYMENTAMOUNT90'])

    den = np.where(den <= 0, epsilon, den)

    df['PAYER_FRAGMENTATION'] = df['PAYERID_CHECK_COUNT'] / den
 
    if 'CHECK_CONTAINS_METLIFE' in df.columns:
        df['METLIFE_EXPOSURE_PROXY'] = df['CHECK_CONTAINS_METLIFE'] * (df['CHECK_PAYMENT_AVG_30'] / df['CHECK_PAYMENT_AVG'].replace(0, 1))
    else:
        df['METLIFE_EXPOSURE_PROXY'] = 0
 
    calls_0_30 = df['PHONECOUNT30']

    calls_30_60 = (df['PHONECOUNT60'] - df['PHONECOUNT30']).clip(lower=0)

    calls_60_90 = (df['PHONECOUNT90'] - df['PHONECOUNT60']).clip(lower=0)

    # df['RWI_PHONE_SCORE'] = ((calls_0_30 * 1.0) + (calls_30_60 * 0.5) + (calls_60_90 * 0.25))

    df['INTERACTION_FRICTION_90'] = (df['PHONECOUNT90'] / df['SUMCHECKPAYMENTCOUNT90'].replace(0, 1))
 
    return df
 
# ==========================================

# 3. CLASSIFICATION BINNING LOGIC

# ==========================================
 
def fit_classification_binner(df, col_name, event_col, n_bins=4, min_bin_samples=500, smoothing_m=100):

    print(f"\n--- Fitting Classification binner for: {col_name} ---")

    global_mean_rate = df[event_col].mean()

    stats = df.groupby(col_name)[event_col].agg(['mean', 'count'])

    stats['smoothed_event_rate'] = (

        (stats['mean'] * stats['count'] + global_mean_rate * smoothing_m) /

        (stats['count'] + smoothing_m)

    )

    rate_map = stats['smoothed_event_rate'].to_dict()

    X_proxy = df[col_name].map(rate_map).fillna(global_mean_rate)

    X_data = X_proxy.to_frame().values.astype(np.float32)

    y_data = df[event_col].fillna(0).values.astype(int)

    binning_tree = DecisionTreeClassifier(

        max_leaf_nodes=n_bins, min_samples_leaf=min_bin_samples, class_weight='balanced', random_state=42

    )

    binning_tree.fit(X_data, y_data)

    df['leaf_id'] = binning_tree.apply(X_data)

    leaf_stats = df.groupby('leaf_id')[event_col].agg(['sum', 'count'])

    leaf_stats['event_rate'] = leaf_stats['sum'] / leaf_stats['count']

    sorted_leaves = leaf_stats.sort_values('event_rate').index.tolist()

    leaf_to_bin_map = {leaf_id: i for i, leaf_id in enumerate(sorted_leaves)}

    binner_artifacts = {

        'col_name': col_name, 'global_mean_rate': global_mean_rate, 'rate_map': rate_map,

        'binning_tree': binning_tree, 'leaf_to_bin_map': leaf_to_bin_map

    }

    print(f"Fit complete for {col_name}. Found {len(sorted_leaves)} bins.")

    return binner_artifacts

def transform_with_binner(df, binner_artifacts, event_col=None):

    col_name = binner_artifacts['col_name']

    rate_map = binner_artifacts['rate_map']

    global_mean_rate = binner_artifacts['global_mean_rate']

    binning_tree = binner_artifacts['binning_tree']

    leaf_to_bin_map = binner_artifacts['leaf_to_bin_map']

    df_new = df.copy()

    prefix = f"{col_name}_BIN"

    X_proxy = df_new[col_name].map(rate_map).fillna(global_mean_rate)

    X_data = X_proxy.to_frame().values.astype(np.float32)

    leaf_ids = pd.Series(binning_tree.apply(X_data), index=X_proxy.index)

    # Get the raw mapped integer (0, 1, 2, 3) or -1 for unknown

    numeric_bins = leaf_ids.map(leaf_to_bin_map).fillna(-1)

    # 🚨 FIX: Leave them as integers! (Add 1 so bins are 1, 2, 3, 4. Unknowns become 0)

    # Because this is an integer, the OneHotEncoder later will ignore it!

    df_new[prefix] = (numeric_bins + 1).astype(int)

    return df_new, None
 
# ==========================================

# 4. DATA PIPELINE WRAPPERS

# ==========================================
 
def run_base_preprocessing(df, time_col):

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

        df['TIME_SINCE_LAST_CANCELLED_CHECK_PAYMENT'].fillna(9999 , inplace=True)

    if 'TIME_SINCE_LAST_CHECK_PAYMENT' in df.columns:

        df['TIME_SINCE_LAST_CHECK_PAYMENT'].fillna(9999, inplace=True)

    df = df[df['SUMCHECKPAYMENTCOUNT'] > 0]

    if time_col in df.columns:

        df = df[df[time_col] >= 90]

    if 'PAYERID_LIST_CHECK' in df.columns:

        df['CHECK_CONTAINS_METLIFE'] = df['PAYERID_LIST_CHECK'].apply(lambda x: int('7376' in str(x)) if pd.notna(x) else 0)

        df['CHECK_IS_ONLY_METLIFE'] = df['PAYERID_LIST_CHECK'].apply(lambda x: int('7376' == str(x)) if pd.notna(x) else 0)

        df['PAYERID_CHECK_COUNT'] = df['PAYERID_LIST_CHECK'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

    df['CHECK_PAYMENT_AVG'] = df['SUMCHECKPAYMENTAMOUNT'] / df['SUMCHECKPAYMENTCOUNT'].replace(0,1)

    df['CHECK_PAYMENT_AVG_30'] = df['SUMCHECKPAYMENTAMOUNT30'] / df['SUMCHECKPAYMENTCOUNT30'].replace(0,1)

    df['CHECK_PAYMENT_AVG_60'] = df['SUMCHECKPAYMENTAMOUNT60'] / df['SUMCHECKPAYMENTCOUNT60'].replace(0,1)

    df['CHECK_PAYMENT_AVG_90'] = df['SUMCHECKPAYMENTAMOUNT90'] / df['SUMCHECKPAYMENTCOUNT90'].replace(0,1)

    if time_col in df.columns:

        df['CHECK_PAYMENTCOUNT_RATE'] = df['SUMCHECKPAYMENTCOUNT'] / df[time_col].replace(0, 1)

        df['CANCELLED_CHECK_PAYMENTCOUNT_RATE'] = df['CANCELLEDCHECKPAYMENTCOUNT30'] / df[time_col].replace(0, 1)

        df['PHONECOUNT_RATE'] = df['PHONECOUNT'] / df[time_col].replace(0, 1)

    df['CHECK_PAYMENTCOUNT_RATE_30'] = df['SUMCHECKPAYMENTCOUNT30'] / 30

    df['CHECK_PAYMENTCOUNT_RATE_60'] = df['SUMCHECKPAYMENTCOUNT60'] / 60

    df['CHECK_PAYMENTCOUNT_RATE_90'] = df['SUMCHECKPAYMENTCOUNT90'] / 90

    df['CANCELLED_CHECK_PAYMENTCOUNT_RATE_30'] = df['CANCELLEDCHECKPAYMENTCOUNT30'] / 30

    df['CANCELLED_CHECK_PAYMENTCOUNT_RATE_60'] = df['CANCELLEDCHECKPAYMENTCOUNT60'] / 60

    df['CANCELLED_CHECK_PAYMENTCOUNT_RATE_90'] = df['CANCELLEDCHECKPAYMENTCOUNT60'] / 90

    if 'SNAPSHOT_END_DATE' in df.columns and 'PROVIDERCREATEDON' in df.columns:

        df['ACCOUNT_AGE'] = (pd.to_datetime(df['SNAPSHOT_END_DATE']) - pd.to_datetime(df['PROVIDERCREATEDON'])).dt.days

    elif 'DATE_END' in df.columns and 'PROVIDERCREATEDON' in df.columns:

        df['ACCOUNT_AGE'] = (pd.to_datetime(df['DATE_END']) - pd.to_datetime(df['PROVIDERCREATEDON'])).dt.days

    if 'PROVIDERCREATEDON' in df.columns and 'DATE_START' in df.columns:

        df['FIRSTACTION_BOOL'] = (df['PROVIDERCREATEDON'] == df['DATE_START']).astype(int)

    if 'SUMIMPORTANTPAYMENTCOUNT' in df.columns:

        df['PCTPAYCOUNTSPONSORED'] = df['SUMPAYERSPONSOREDPAYMENTCOUNT'] / df['SUMIMPORTANTPAYMENTCOUNT'].replace(0,1)

        df['PCTPAYAMOUNTSPONSORED'] = df['SUMPAYERSPONSOREDPAYMENTAMOUNT'] / df['SUMIMPORTANTPAYMENTAMOUNT'].replace(0,1)

    df['CHECK_PAYMENT_AVG_30'].fillna(0, inplace = True)

    if 'CANCELLED_CHECK_PAYMENTCOUNT_RATE' in df.columns:

        df['CANCELLEDCHECK_PAYMENTCOUNT_MOMENTUM_30_ALLTIME'] = df['CANCELLED_CHECK_PAYMENTCOUNT_RATE_30'] / df['CANCELLED_CHECK_PAYMENTCOUNT_RATE'].replace(0, 0.001)

        df['CANCELLEDCHECK_PAYMENTCOUNT_MOMENTUM_30_ALLTIME'].fillna(0, inplace = True)

    df['CANCELLEDCHECK_PAYMENTCOUNT_MOMENTUM_30_90'] = df['CANCELLED_CHECK_PAYMENTCOUNT_RATE_30'] / df['CANCELLED_CHECK_PAYMENTCOUNT_RATE_90'].replace(0, 0.001)

    df['CANCELLEDCHECK_PAYMENTCOUNT_MOMENTUM_30_90'].fillna(-1, inplace = True)

    if 'CHECK_PAYMENTCOUNT_RATE' in df.columns:

        df['CHECK_PAYMENTCOUNT_MOMENTUM_30_ALLTIME'] = df['CHECK_PAYMENTCOUNT_RATE_30'] / df['CHECK_PAYMENTCOUNT_RATE'].replace(0, 0.001)

    df['CHECK_PAYMENTCOUNT_MOMENTUM_30_90'] = df['CHECK_PAYMENTCOUNT_RATE_30'] / df['CHECK_PAYMENTCOUNT_RATE_90'].replace(0, 0.001)

    df['CHECK_PAYMENTCOUNT_MOMENTUM_30_90'].fillna(-1, inplace = True)

    df['CHECK_PAYMENTCOUNT_MOMENTUM_30_60'] = df['CHECK_PAYMENTCOUNT_RATE_30'] / df['CHECK_PAYMENTCOUNT_RATE_60'].replace(0, 0.001)

    df['CHECK_PAYMENTCOUNT_MOMENTUM_30_60'].fillna(0, inplace = True)

    df['PHONECOUNT_RATE_30'] = df['PHONECOUNT30'] / 30

    df['PHONECOUNT_RATE_60'] = df['PHONECOUNT60'] / 60

    df['PHONECOUNT_RATE_90'] = df['PHONECOUNT90'] / 60

    if 'PHONECOUNT_RATE' in df.columns:

        df['PHONECOUNT_MOMENTUM_30_ALLTIME'] = df['PHONECOUNT_RATE_30'] / df['PHONECOUNT_RATE'].replace(0, 0.001)

        df['PHONECOUNT_MOMENTUM_30_ALLTIME'].fillna(0, inplace = True)

    df['PHONECOUNT_MOMENTUM_30_60'] = df['PHONECOUNT_RATE_30'] / df['PHONECOUNT_RATE_60'].replace(0, 0.001)

    df['PHONECOUNT_MOMENTUM_30_60'].fillna(0, inplace = True)

    df['PHONECOUNT_MOMENTUM_30_90'] = df['PHONECOUNT_RATE_30'] / df['PHONECOUNT_RATE_90'].replace(0, 0.001)

    df['PHONECOUNT_MOMENTUM_30_90'].fillna(0, inplace = True)

    df.reset_index(drop = True, inplace = True)

    if 'PROVIDERTYPE' in df.columns: df['PROVIDERTYPE'] = df['PROVIDERTYPE'].astype('category')

    if 'SEGMENT' in df.columns: df['SEGMENT'] = df['SEGMENT'].astype('category')

    if 'ACCOUNT_AGE' in df.columns:

        bins = [-float('inf'), 730, 1460, 2190, 2920, 3650, 4380, float('inf')]

        labels = ['LT2YR_OLD', 'BT2_4YR_OLD', 'BT4_6YR_OLD', 'BT6_8YR_OLD', 'BT8_10YR_OLD', 'BT10_12YR_OLD', 'GT12YR_OLD']

        df['ACCOUNT_AGE_BIN'] = pd.cut(df['ACCOUNT_AGE'], bins=bins, labels=labels, right=True, include_lowest=True)

    df = PHONE_CALL_DENSITY(df, ['',30,60,90])

    df = Cancelled_Check_Amount_Ratio(df, ['',30,60,90])

    df = CHECK_PAYMENT_STABILITY(df, [30,60,90])

    df = LONG_STARTER_BOOL(df, 365, time_col)

    df = independent_features(df, time_col)

    if 'STATE' in df.columns and 'TAXONOMYGENERAL' in df.columns:

        df['STATE_TAX_INTERACT'] = df['STATE'].astype(str) + '_' + df['TAXONOMYGENERAL'].astype(str)

    df['SUMCHECKPAYMENTCOUNT30_BOOL'] = df['SUMCHECKPAYMENTCOUNT30'].apply(lambda x: 1 if x > 0 else 0)

    return df
 
def preprocess_train(df):

    df = run_base_preprocessing(df, TRAIN_TIME_COL)

    state_binner_artifacts = fit_classification_binner(df, 'STATE', TRAIN_TARGET_COL)

    tax_binner_artifacts = fit_classification_binner(df, 'TAXONOMYGENERAL', TRAIN_TARGET_COL)

    state_tax_binner_artifacts = fit_classification_binner(df, 'STATE_TAX_INTERACT', TRAIN_TARGET_COL, n_bins=5)

    df, _ = transform_with_binner(df, state_binner_artifacts, TRAIN_TARGET_COL)

    df, _ = transform_with_binner(df, tax_binner_artifacts, TRAIN_TARGET_COL)

    df, _ = transform_with_binner(df, state_tax_binner_artifacts, TRAIN_TARGET_COL)

    return df, state_binner_artifacts, tax_binner_artifacts, state_tax_binner_artifacts
 
def preprocess_oos(df, state_binner_artifacts, tax_binner_artifacts, state_tax_binner_artifacts):

    df = run_base_preprocessing(df, OOS_TIME_COL)

    df, _ = transform_with_binner(df, state_binner_artifacts)

    df, _ = transform_with_binner(df, tax_binner_artifacts)

    df, _ = transform_with_binner(df, state_tax_binner_artifacts)

    return df


def preprocess_val(df, state_binner_artifacts, tax_binner_artifacts, state_tax_binner_artifacts):

    """Validation preprocessing: same flow as OOS — apply train-fit binners only."""

    df = run_base_preprocessing(df, TRAIN_TIME_COL)

    df, _ = transform_with_binner(df, state_binner_artifacts)

    df, _ = transform_with_binner(df, tax_binner_artifacts)

    df, _ = transform_with_binner(df, state_tax_binner_artifacts)

    return df
 
# ==========================================

# 5. LEAK-PROOF SPLITTING & ENCODING (AUTOMATED)

# ==========================================
 
def execute_split_and_feature_selection(df_train_raw, df_val_raw, df_oos_raw):

    df_processed_train = df_train_raw.copy()
    df_processed_val   = df_val_raw.copy()

    # Base feature list

    selected_features = [

        'TIN', 'PROVIDERTYPE', 'SEGMENT', 'SUMCHECKPAYMENTCOUNT30', 'SUMCHECKPAYMENTAMOUNT30', 'CHECK_PAYMENT_AVG_30',

        'CHECK_PAYMENTCOUNT_MOMENTUM_30_90', 'PAYERID_CHECK_COUNT', 'TIME_SINCE_LAST_CHECK_PAYMENT',

        'CANCELLEDCHECKPAYMENTCOUNT30', 'TIME_SINCE_LAST_CANCELLED_CHECK_PAYMENT',  'CHECK_CONTAINS_METLIFE',

        'FIRSTACTION_BOOL', 'PREVIOUSACHOPTINCOUNT', 'ACCOUNT_AGE', 'ACCOUNT_AGE_BIN', TRAIN_TIME_COL,

        TRAIN_TARGET_COL, 'CANCELLED_CHECK_AMOUNT_RATIO30', 'CHECK_PAYMENT_STABILITY_30_60',

        'LONG_UNDECIDED_BOOL', 'ACH_PREFERENCE_RATIO', 'Avg_Days_Between_Check_Payments', 'ENGAGEMENT_ADJUSTED_FRICTION',

        'TAXONOMYGENERAL_BIN', 'STATE_BIN', 'PHONE_CALL_DENSITY_90', 'STATUS_RECENCY_GAP', 'STATE_TAX_INTERACT_BIN',

        'ACTIVE_FRUSTRATION', 'CHECK_DENSITY', 'UNDECIDED_HIGH_ROLLER', 'CANCEL_SHOCK', 'RECENCY_SEVERITY',

        'PAPER_FATIGUE', 'REVENUE_VELOCITY', 'PAIN_POINT_RATIO', 'PAYMENT_ACCELERATION', 'MOMENTUM_30_LIFETIME',

        'PAYMENT_VOLATILITY_INDEX', 'CHECK_FAILURE_RATE_90', 'PAYER_FRAGMENTATION', 'METLIFE_EXPOSURE_PROXY',

        'RWI_PHONE_SCORE', 'INTERACTION_FRICTION_90'

    ]

    # Ensure Core Columns Exist

    if 'TIN' not in selected_features: selected_features.insert(0, 'TIN')

    if TRAIN_TARGET_COL not in selected_features: selected_features.append(TRAIN_TARGET_COL)
 
    available_train_features = [f for f in selected_features if f in df_processed_train.columns]

    print(f"\n✅ Automatically processing {len(available_train_features)} engineered features.")

    # 1. NO INTERNAL SPLIT — train/val/oos are already date-separated upstream.
    #    Just filter each to the selected features. Train and Val use the same
    #    feature names; OOS column-aligns below via the masking trick.

    df_train = df_processed_train[available_train_features].reset_index(drop=True)

    df_val = df_processed_val[[f for f in available_train_features if f in df_processed_val.columns]].reset_index(drop=True)

    # 2. Map OOS variables and align column names precisely

    oos_feat_list = []

    for f in available_train_features:

        if f == TRAIN_TARGET_COL:

            oos_feat_list.append(OOS_TARGET_COL)

        elif f == TRAIN_TIME_COL:

            oos_feat_list.append(OOS_TIME_COL)

        else:

            oos_feat_list.append(f)

    df_oos = df_oos_raw[[f for f in oos_feat_list if f in df_oos_raw.columns]].reset_index(drop=True)
 
    # 🚨 MASKING TRICK 🚨 Temporarily rename OOS columns so Scikit-Learn doesn't crash looking for Train names

    df_oos_masked = df_oos.rename(columns={OOS_TARGET_COL: TRAIN_TARGET_COL, OOS_TIME_COL: TRAIN_TIME_COL})
 
    # 3. Identify Numeric and Categorical columns to transform

    exclude_cols = ['TIN', TRAIN_TARGET_COL]

    feature_cols_only = [c for c in df_train.columns if c not in exclude_cols]

    num_cols = list(df_train[feature_cols_only].select_dtypes(include=[np.number]).columns)

    cat_cols = list(set(feature_cols_only) - set(num_cols))
 
    # 4. Initialize Encoder (No Scaler)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
 
    # 5. Processing Helper Function

    def encode_only(df_subset, is_train=False):

        target_col = TRAIN_TARGET_COL 

        if cat_cols:

            if is_train:

                encoded_cats = encoder.fit_transform(df_subset[cat_cols])

            else:

                encoded_cats = encoder.transform(df_subset[cat_cols])

            df_cat = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))

        else:

            df_cat = pd.DataFrame()
 
        df_num = df_subset[num_cols].reset_index(drop=True)
 
        df_final = pd.concat([df_subset[['TIN', target_col]], df_cat, df_num], axis=1)

        df_final = reduce_mem_usage(df_final)

        return df_final
 
    print("\n--- Encoding TRAIN Data ---")

    train_data_final = encode_only(df_train, is_train=True)

    print("\n--- Encoding VALIDATION Data ---")

    val_data_final = encode_only(df_val, is_train=False)

    print("\n--- Encoding OOS Data ---")

    # Pass the masked dataframe into the encoder

    oos_data_final = encode_only(df_oos_masked, is_train=False)
 
    # 🚨 UNMASKING TRICK 🚨 Rename columns back to their original OOS names before returning

    oos_data_final = oos_data_final.rename(columns={TRAIN_TARGET_COL: OOS_TARGET_COL, TRAIN_TIME_COL: OOS_TIME_COL})
 
    return train_data_final, val_data_final, oos_data_final
 
# ==========================================

# 6. MAIN EXECUTION

# ==========================================
 
def main():

    try:

        _catalog = load_model_catalog()
        bucket_name, _folders = get_s3_config(_catalog)

        s3_folders = {

            "new_train": _folders["raw_train"],

            "new_val":   _folders["raw_val"],

            "new_oos":   _folders["raw_oos"],

        }

        print('Fetching Data From S3...')

        new_train_file = get_latest_file_from_s3(bucket_name, s3_folders["new_train"])

        new_val_file   = get_latest_file_from_s3(bucket_name, s3_folders["new_val"])

        new_oos_file   = get_latest_file_from_s3(bucket_name, s3_folders["new_oos"])

        new_train_df = read_csv_from_s3(bucket_name, new_train_file)

        new_val_df   = read_csv_from_s3(bucket_name, new_val_file)

        new_oos_df   = read_csv_from_s3(bucket_name, new_oos_file)

        print('Preprocessing Started...')

        df_train_processed, state_binner, tax_binner, state_tax_binner = preprocess_train(new_train_df)

        df_val_processed = preprocess_val(new_val_df, state_binner, tax_binner, state_tax_binner)

        df_oos_processed = preprocess_oos(new_oos_df, state_binner, tax_binner, state_tax_binner)

        print('Mathematical Preprocessing Completed.')

        train_data_final, val_data_final, oos_data_final = execute_split_and_feature_selection(df_train_processed, df_val_processed, df_oos_processed)

        print("\n--- Saving Final Outputs to S3 ---")

        Train_folder = _folders["preprocessed_train"]

        Val_folder   = _folders["preprocessed_val"]

        OOS_folder   = _folders["preprocessed_oos"]

        write_csv_to_s3(train_data_final, bucket_name, Train_folder, "train_data_new")

        write_csv_to_s3(val_data_final, bucket_name, Val_folder, "val_data_new")

        write_csv_to_s3(oos_data_final, bucket_name, OOS_folder, "oos_data_new")

        print("✅ Data successfully engineered, encoded, memory-optimized, and saved to S3.")
 
    except Exception as e:

        print(f"\n❌ Error in pipeline: {e}")

        return False
 
if __name__ == "__main__":

    main()
 