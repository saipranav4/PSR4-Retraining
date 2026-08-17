import boto3
import pandas as pd
import numpy as np
import pickle
import json

from io import BytesIO
from datetime import datetime
from InquirerPy import inquirer
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==============================================================
# >>> CHANGE 1: NEW IMPORT FOR UNDERSAMPLING <<<
# ==============================================================
from imblearn.under_sampling import RandomUnderSampler
# ==============================================================
# <<< END CHANGE 1 >>>
# ==============================================================

# Local Imports
from model_registry import MODEL_REGISTRY
from catalog_loader import load_model_catalog, get_s3_config
from hyperparameter_tuning import get_param_distributions, tune_model

# ==============================================================
# >>> CHANGE 2: NEW CONFIG — UNDERSAMPLING RATIO <<<
# ==============================================================
UNDERSAMPLE_RATIO = 200    # 1 positive per N negatives. Set to None to skip.
# ==============================================================
# <<< END CHANGE 2 >>>
# ==============================================================

# =========================================================
# S3 Utilities
# =========================================================

def get_latest_file_from_s3(bucket_name, prefix):
    s3 = boto3.client("s3")
    print(f"\n🔎 Listing objects in bucket='{bucket_name}' with prefix='{prefix}'...")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" not in response:
        raise FileNotFoundError(f"No files found in S3 with prefix: {prefix}")
    files = sorted([f for f in response["Contents"] if not f["Key"].endswith('/')], key=lambda x: x["LastModified"], reverse=True)
    latest_key = files[0]["Key"]
    print(f"✅ Latest file for prefix '{prefix}' is: {latest_key}")
    return latest_key


def read_csv_from_s3(bucket_name, s3_key):
    s3 = boto3.client("s3")
    print(f"📥 Downloading CSV from s3://{bucket_name}/{s3_key}")
    response = s3.get_object(Bucket=bucket_name, Key=s3_key)
    return pd.read_csv(BytesIO(response["Body"].read()))


def save_model_to_s3(model, bucket_name, folder, filename):
    s3 = boto3.client("s3")
    s3_key = f"{folder}{filename}"
    print(f"📤 Uploading model to s3://{bucket_name}/{s3_key}")
    pickle_buffer = BytesIO()
    pickle.dump(model, pickle_buffer)
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=pickle_buffer.getvalue())
    return s3_key


def save_metadata_to_s3(metadata_dict, bucket_name, folder, filename):
    s3 = boto3.client("s3")
    s3_key = f"{folder}{filename}"
    print(f"📤 Uploading metadata to s3://{bucket_name}/{s3_key}")
    json_str = json.dumps(metadata_dict, indent=4)
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=json_str)
    return s3_key

# =========================================================
# Main Execution
# =========================================================

def main():
    print("==============================================")
    print("🚀 STARTING CLASSIFICATION MODEL BUILD PIPELINE")
    print("==============================================\n")

    _catalog = load_model_catalog()
    bucket_name, _folders = get_s3_config(_catalog)
    train_folder      = _folders["preprocessed_train"]
    val_folder        = _folders["preprocessed_val"]
    model_save_folder = _folders["model_output"]

    # ---------------------------
    # 1. Load Data (Train & Validation)
    # ---------------------------
    try:
        train_key = get_latest_file_from_s3(bucket_name, train_folder + "train_data_new")
        val_key   = get_latest_file_from_s3(bucket_name, val_folder + "val_data_new")
        train_df = read_csv_from_s3(bucket_name, train_key)
        val_df   = read_csv_from_s3(bucket_name, val_key)
    except Exception as e:
        print(f"❌ Failed to load Data from S3: {e}")
        return

    # ---------------------------
    # 2. Interactive Column Selection
    # ---------------------------
    all_columns = list(train_df.columns)
    selected_target = 'OPT_IN_ACH'
    print(f"\n🎯 Target variable automatically set to: {selected_target}")

    feature_candidates = [c for c in all_columns if c not in [selected_target, 'TIN']]
    selected_features = inquirer.checkbox(
        message="Select the FEATURES to train on (Space to select, Enter to confirm):",
        choices=feature_candidates,
        default=feature_candidates,
    ).execute()

    if not selected_features:
        print("❌ No features selected. Exiting.")
        return

    # ---------------------------
    # 3. Prepare X and y Matrices
    # ---------------------------
    print("\nPreparing X and y matrices...")
    X_train = train_df[selected_features]
    y_train = train_df[selected_target].astype(int)
    X_val   = val_df[selected_features]
    y_val   = val_df[selected_target].astype(int)

    # ==============================================================
    # >>> CHANGE 3: UNDERSAMPLE TRAINING DATA ONLY <<<
    # ==============================================================
    # Applies AFTER train/val are loaded, BEFORE imbalance ratio
    # is computed. Validation data is NEVER touched.
    # ==============================================================
    if UNDERSAMPLE_RATIO is not None:
        print(f"\n📉 Undersampling training data to ratio 1:{UNDERSAMPLE_RATIO}...")
        print(f"   Before: {len(y_train):,} rows | {y_train.sum():,} positives ({y_train.mean()*100:.4f}%)")

        n_pos = int(y_train.sum())
        n_neg = n_pos * UNDERSAMPLE_RATIO

        rus = RandomUnderSampler(
            sampling_strategy={0: n_neg, 1: n_pos},
            random_state=42
        )
        X_train, y_train = rus.fit_resample(X_train, y_train)

        print(f"   After:  {len(y_train):,} rows | {y_train.sum():,} positives ({y_train.mean()*100:.4f}%)")
    else:
        print("\n📉 Skipping undersampling (UNDERSAMPLE_RATIO is None)")
    # ==============================================================
    # <<< END CHANGE 3 >>>
    # ==============================================================

    # Dynamic Weights Calculation
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1
    print(f"   Target Distribution: {neg_count} Negatives | {pos_count} Positives")
    print(f"   Calculated Imbalance Ratio (scale_pos_weight): {imbalance_ratio:.2f}")

    # ---------------------------
    # 4. Model Registry Selection
    # ---------------------------
    catalog = load_model_catalog()
    project_name = list(catalog.get("production_models", {}).keys())[0]
    available_models = list(MODEL_REGISTRY.get("classification", {}).keys())

    selected_model_names = inquirer.checkbox(
        message="Select the classification models to train:",
        choices=available_models,
        default=["RandomForestClassifier", "XGBClassifier", "LogisticRegression"]
    ).execute()

    # ---------------------------
    # 5. Save Selection Metadata
    # ---------------------------
    metadata = {
        "project_name": project_name,
        "selected_features": selected_features,
        "target": selected_target,
        "selected_models": selected_model_names,
        "imbalance_ratio": imbalance_ratio,
        # ==============================================================
        # >>> CHANGE 4: TRACK UNDERSAMPLE RATIO IN METADATA <<<
        # ==============================================================
        "undersample_ratio": UNDERSAMPLE_RATIO,
        # ==============================================================
        # <<< END CHANGE 4 >>>
        # ==============================================================
        "training_timestamp": str(datetime.utcnow())
    }
    meta_filename = f"{project_name}__selection_metadata__{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    save_metadata_to_s3(metadata, bucket_name, model_save_folder, meta_filename)

    # ---------------------------
    # 6. Train & Tune Models
    # ---------------------------
    results = {}
    models_needing_scaling = ["SVC", "KNeighborsClassifier", "LogisticRegression", "GaussianNB"]

    for model_name in tqdm(selected_model_names, desc="Training Models"):
        print(f"\n======================================")
        print(f"🚀 Training: {model_name}")
        print(f"======================================")

        try:
            model_class = MODEL_REGISTRY["classification"][model_name]

            # --- Inject Dynamic Imbalance Weights & Core Params ---
            init_params = {}
            if model_name in ["XGBClassifier", "LGBMClassifier"]:
                init_params["scale_pos_weight"] = imbalance_ratio
                if model_name == "LGBMClassifier":
                    init_params["verbosity"] = -1
                elif model_name == "XGBClassifier":
                    init_params["verbosity"] = 0
            elif model_name in ["RandomForestClassifier", "DecisionTreeClassifier", "LogisticRegression", "SVC"]:
                init_params["class_weight"] = "balanced"

            if model_name == "XGBClassifier":
                init_params["early_stopping_rounds"] = 20

            if model_name == "SVC":
                init_params["probability"] = True

            base_model = model_class(**init_params)

            # --- Inject Scaling Pipeline if needed ---
            if model_name in models_needing_scaling:
                print(f"   [!] {model_name} requires scaling. Wrapping in StandardScaler Pipeline.")
                model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', base_model)
                ])
            else:
                model = base_model

            # --- DYNAMIC ITERATION LOGIC ---
            if model_name in ["XGBClassifier", "CatBoostClassifier", "LGBMClassifier"]:
                custom_iter = 20
            elif model_name in ["RandomForestClassifier", "ExtraTreesClassifier"]:
                custom_iter = 5
            else:
                custom_iter = 10

            # --- Tuning & Fitting ---
            param_dist = get_param_distributions(model_name)
            best_model, best_params, tuning_log = tune_model(
                model_name=model_name,
                model=model,
                param_distributions=param_dist,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                n_iter=custom_iter
            )

            # --- Save model artifact ---
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"{project_name}__{model_name}__{timestamp}.pkl"
            model_artifact = {
                "model": best_model,
                "features": selected_features,
                "tuning_log": tuning_log
            }
            s3_key = save_model_to_s3(model_artifact, bucket_name, model_save_folder, filename)
            results[model_name] = {"s3_key": s3_key, "best_params": best_params}

        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    print("\n==============================")
    print("✅ MODEL BUILDING COMPLETED")
    print("==============================\n")


if __name__ == "__main__":
    main()
