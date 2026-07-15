import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit


def get_param_distributions(model_name):
    """
    Return the hyperparameter search space for a given model family.
    Called by tune_model() to build the RandomizedSearchCV grid.
    """

    if "RandomForest" in model_name or "ExtraTrees" in model_name:
        return {
            "n_estimators": [58],
            "max_depth": [8],
            "min_samples_split": [19956],
            "min_samples_leaf": [25529],
            "max_features": ["sqrt", "log2"],
        }

    elif "DecisionTree" in model_name:
        return {
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [50, 100, 500],
            "min_samples_leaf": [50, 100, 500, 1000],
        }

    elif "XGB" in model_name or "GradientBoosting" in model_name or "LGBM" in model_name or "CatBoost" in model_name:
        return {
                   "n_estimators":      [608],
                   "max_depth":         [3],
                   "learning_rate":     [0.087],
                   "min_child_weight":  [1],
                   "reg_alpha":         [0.349],
                   "reg_lambda":        [34.5],
                   "gamma":             [2.49],
                   "subsample":         [0.764],
                   "colsample_bytree":  [0.902],
                   "colsample_bynode":  [0.691],
                   "scale_pos_weight":  [355],

        }

    elif model_name in ["SVC", "SVR"]:
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__kernel": ["linear", "rbf"],
        }

    elif "KNeighbors" in model_name:
        return {
            "classifier__n_neighbors": [5, 10, 50, 100],
            "classifier__weights": ["uniform", "distance"],
        }

    elif "LogisticRegression" in model_name:
        return {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
        }

    elif "AdaBoost" in model_name:
        return {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1, 1.0],
        }

    return {}


def tune_model(model_name, model, param_distributions, X_train, y_train,
               X_val=None, y_val=None, n_iter=15):
    """
    Tune the model via RandomizedSearchCV optimizing PR-AUC (average_precision).

    - Uses a fast single-fold StratifiedShuffleSplit (test_size=0.2) so search runs
      quickly on the undersampled training set.
    - Enables early stopping via the external Validation set for boosting models
      (XGBoost, LightGBM, CatBoost).
    - Falls back to a plain model.fit() if the search itself errors out.

    Returns:
        best_estimator     — the tuned model
        best_params        — dict of best hyperparameters (None if no grid)
        log_df             — tuning history DataFrame (None if no grid or on error)
    """

    if not param_distributions:
        print("No hyperparameter grid defined. Fitting base model.")
        model.fit(X_train, y_train)
        return model, None, None

    fit_params = {}
    boosting_models = ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]

    if X_val is not None and y_val is not None and model_name in boosting_models:
        print("   [!] Enabling Early Stopping with External Validation Set.")
        fit_params['eval_set'] = [(X_val, y_val)]

        if model_name == "LGBMClassifier":
            from lightgbm import early_stopping
            fit_params['callbacks'] = [early_stopping(stopping_rounds=20, verbose=False)]
        else:
            # Force XGBoost and CatBoost to be perfectly silent during training
            fit_params['verbose'] = False
            if model_name == "CatBoostClassifier":
                fit_params['early_stopping_rounds'] = 20

    try:
        fast_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring='average_precision',   # PR-AUC — better for rare-event problems
            cv=fast_cv,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )

        print(f"   Starting RandomizedSearchCV (Optimizing for PR-AUC, Iterations: {n_iter})...")
        search.fit(X_train, y_train, **fit_params)

        print(f"   Best parameters found: {search.best_params_}")
        print(f"   Best Cross-Validation PR-AUC: {search.best_score_:.4f}")

        # Extract Tuning History into a clean DataFrame
        results_df = pd.DataFrame(search.cv_results_)
        log_df = results_df[['params', 'mean_test_score', 'rank_test_score', 'mean_fit_time']].copy()
        log_df.rename(columns={
            'mean_test_score': 'PR_AUC_Score',
            'rank_test_score': 'Rank',
            'mean_fit_time': 'Train_Time_s',
        }, inplace=True)
        log_df.sort_values(by='Rank', inplace=True)
        log_df.reset_index(drop=True, inplace=True)

        return search.best_estimator_, search.best_params_, log_df

    except Exception as e:
        print(f"   ⚠️ Tuning failed: {e}. Falling back to default parameters.")
        if fit_params:
            model.fit(X_train, y_train, **fit_params)
        else:
            model.fit(X_train, y_train)
        return model, None, None
