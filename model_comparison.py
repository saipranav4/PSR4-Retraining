import pandas as pd

import numpy as np

import pickle

import boto3

import io

import os

import random

import warnings

import json

from typing import Dict, Any, List

from io import BytesIO

from datetime import datetime
 
# SHAP & Visualization Imports

import shap

import matplotlib.pyplot as plt
 
# Sklearn Classification Imports

from sklearn.metrics import (accuracy_score, precision_score, recall_score, 

                             f1_score, roc_auc_score, classification_report, confusion_matrix)
 
# ReportLab Imports

from reportlab.lib import colors

from reportlab.lib.pagesizes import letter

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

from reportlab.lib.units import inch
 
# Local Module Imports

try:

    from catalog_loader import load_model_catalog, get_s3_config

except ImportError:

    pass
 
warnings.filterwarnings('ignore')

random.seed(42)
 
# ==========================================

# 0. GLOBAL CONFIGURATION & S3 CLIENT

# ==========================================

TARGET_COLUMN = 'OPT_IN_ACH'

SEGMENT_COLUMN = 'FIRSTACTION_BOOL'

TARGET_COLUMN_OOS = 'OPT_IN_ACH'
 
s3_client = boto3.client('s3')
 
# ==========================================

# 1. ORIGINAL DECILE & CUTOFF FUNCTIONS

# ==========================================
 
def create_quantile_table(y_prob, y_train, splits=10):

    data={'PROB':y_prob,'TARGET':y_train}

    table=pd.DataFrame(data)

    table.sort_values(by=['PROB'],inplace=True,ascending=False)

    target_count = table['TARGET'].sum() 

    list_df = np.array_split(table, splits)
 
    total_count_list, target_count_list, target_percent_list = [], [], []

    min_prob_yes, max_prob_yes, mean_prob_yes, median_prob_yes, std_prob_yes = [], [], [], [], []

    Overall_Dispute_Rate=[]
 
    for split_df in list_df:

        target_sum = split_df['TARGET'].sum()

        num_rows = split_df.shape[0]

        target_percent = (target_sum / num_rows) * 100 if num_rows > 0 else 0

        target_percent_list.append(target_percent)

        target_count_list.append(target_sum)

        total_count_list.append(num_rows)

        Overall_Dispute_Rate.append((target_sum / target_count) * 100 if target_count > 0 else 0)
 
        min_prob_yes.append(split_df['PROB'].min())

        max_prob_yes.append(split_df['PROB'].max())

        mean_prob_yes.append(split_df['PROB'].mean())

        median_prob_yes.append(split_df['PROB'].median())

        std_prob_yes.append(split_df['PROB'].std())
 
    table_df = pd.DataFrame({

        'Counts': total_count_list,

        'Positive Target': target_count_list,

        'Target Rate': target_percent_list,

        'Overall_Target PCT':Overall_Dispute_Rate,

        'Min PROB_OF_YES': min_prob_yes,

        'Max PROB_OF_YES': max_prob_yes,

        'Mean PROB_OF_YES': mean_prob_yes,

        'Median PROB_OF_YES': median_prob_yes,

        'Std PROB_OF_YES': std_prob_yes

    })

    return table_df
 
def get_cutoffs(decile_table):

    cutoffs=[]

    for i in range(len(decile_table['Min PROB_OF_YES'])-1):

        value_1=decile_table['Min PROB_OF_YES'].iloc[i]

        value_2=decile_table['Min PROB_OF_YES'].iloc[i+1]

        if i==0:

            cutoffs.append((value_1,10))

            cutoffs.append((value_2,value_1))

        elif i==8:

            cutoffs.append((0,value_1))

        else:

            cutoffs.append((value_2,value_1))

    return cutoffs
 
def get_cutoffs_ventile(decile_table):

    cutoffs=[]

    for i in range(len(decile_table['Min PROB_OF_YES'])-1):

        value_1=decile_table['Min PROB_OF_YES'].iloc[i]

        value_2=decile_table['Min PROB_OF_YES'].iloc[i+1]

        if i==0:

            cutoffs.append((value_1,10))

            cutoffs.append((value_2,value_1))

        elif i==18:

            cutoffs.append((0,value_1))

        else:

            cutoffs.append((value_2,value_1))

    return cutoffs
 
def get_cutoffs_generic(decile_table):

    cutoffs=[]

    n_rows = len(decile_table['Min PROB_OF_YES'])

    for i in range(n_rows-1):

        value_1=decile_table['Min PROB_OF_YES'].iloc[i]

        value_2=decile_table['Min PROB_OF_YES'].iloc[i+1]

        if i==0:

            cutoffs.append((value_1,10))

            cutoffs.append((value_2,value_1))

        elif i==n_rows-2:

            cutoffs.append((0,value_1))

        else:

            cutoffs.append((value_2,value_1))

    return cutoffs
 
def create_valid_decile_analysis(cutoffs, proba_of_interest, optin_of_interest, decile_table, total_count):

    count, disputed_counts, one_percents = [], [], []

    min_prob_yes, max_prob_yes, mean_prob_yes, median_prob_yes, std_prob_yes = [], [], [], [], []

    Overall_Dispute_Rate=[]

    for i in range(len(cutoffs)):

        lower_bound=cutoffs[i][0]

        upper_bound=cutoffs[i][1]

        split_df = decile_table.query(f'{proba_of_interest} >= {lower_bound} and {proba_of_interest} <{upper_bound}')

        ones = split_df[f'{optin_of_interest}'].sum()

        num_rows = split_df.shape[0]

        one_percents.append((ones / num_rows) * 100 if num_rows > 0 else 0)

        disputed_counts.append(ones)

        count.append(num_rows)

        Overall_Dispute_Rate.append((ones /total_count ) * 100 if total_count > 0 else 0)

        min_prob_yes.append(split_df[f'{proba_of_interest}'].min())

        max_prob_yes.append(split_df[f'{proba_of_interest}'].max())

        mean_prob_yes.append(split_df[f'{proba_of_interest}'].mean())

        median_prob_yes.append(split_df[f'{proba_of_interest}'].median())

        std_prob_yes.append(split_df[f'{proba_of_interest}'].std())

    decile_analysis_df = pd.DataFrame({

        'Counts': count,

        'Positive Target': disputed_counts,

        'Target Rate': one_percents,

        'Overall_Target PCT':Overall_Dispute_Rate,

        'Min PROB_OF_YES': min_prob_yes,

        'Max PROB_OF_YES': max_prob_yes,

        'Mean PROB_OF_YES': mean_prob_yes,

        'Median PROB_OF_YES': median_prob_yes,

        'Std PROB_OF_YES': std_prob_yes

    })

    return decile_analysis_df

def create_account_age_table(df, target_col, prob_array):
    """
    Account age breakdown using the 7-bucket scheme from feature_engineering.py.
    Returns a table with the same columns as create_quantile_table.
    """
    bins = [-float('inf'), 730, 1460, 2190, 2920, 3650, 4380, float('inf')]
    labels = ['LT2YR_OLD', 'BT2_4YR_OLD', 'BT4_6YR_OLD', 'BT6_8YR_OLD',
              'BT8_10YR_OLD', 'BT10_12YR_OLD', 'GT12YR_OLD']

    work_df = pd.DataFrame({
        'ACCOUNT_AGE': df['ACCOUNT_AGE'].values,
        'TARGET': df[target_col].astype(int).values,
        'PROB': prob_array,
    })
    work_df['_AGE_BIN'] = pd.cut(work_df['ACCOUNT_AGE'], bins=bins, labels=labels,
                                  right=True, include_lowest=True)

    total_positives = work_df['TARGET'].sum()
    total_count = len(work_df)
    overall_rate = (total_positives / total_count) * 100 if total_count > 0 else 0

    rows = []
    for label in labels:
        bucket = work_df[work_df['_AGE_BIN'] == label]
        cnt = len(bucket)
        pos = int(bucket['TARGET'].sum())
        target_rate = (pos / cnt * 100) if cnt > 0 else 0
        overall_pct = (pos / total_positives * 100) if total_positives > 0 else 0
        lift = (target_rate / overall_rate) if overall_rate > 0 else 0

        rows.append({
            'Bin': label,
            'Counts': cnt,
            'Positive Target': pos,
            'Target Rate': target_rate,
            'Overall_Target PCT': overall_pct,
            'Min PROB_OF_YES': bucket['PROB'].min() if cnt > 0 else 0,
            'Max PROB_OF_YES': bucket['PROB'].max() if cnt > 0 else 0,
            'Mean PROB_OF_YES': bucket['PROB'].mean() if cnt > 0 else 0,
            'Median PROB_OF_YES': bucket['PROB'].median() if cnt > 0 else 0,
            'Std PROB_OF_YES': bucket['PROB'].std() if cnt > 0 else 0,
            'Lift': lift,
        })

    return pd.DataFrame(rows)

# ==========================================

# 2. GRANULAR EXCEL EXPORT

# ==========================================

def export_granular_excel(filename, overall_data, cb_data, und_data,
                          age_overall_data=None, age_cb_data=None, age_und_data=None):

    try:

        with pd.ExcelWriter(filename, engine="openpyxl") as writer:

            def write_segment_list_to_sheet(data_list, sheet_name,
                                             left_label="TRAIN",
                                             middle_label="VALIDATION",
                                             right_label="OOS (TEST)"):

                writer.book.create_sheet(sheet_name)

                worksheet = writer.book[sheet_name]

                writer.sheets[sheet_name] = worksheet

                current_row = 0

                for title, left_df, middle_df, right_df in data_list:

                    pd.DataFrame([f"SEGMENT: {title}"]).to_excel(

                        writer, sheet_name=sheet_name, startrow=current_row, startcol=0, header=False, index=False

                    )

                    current_row += 1

                    # Compute column offsets so the 3 tables sit side by side with 2 blank cols between them.

                    left_cols   = len(left_df.columns)   if left_df   is not None else 9

                    middle_cols = len(middle_df.columns) if middle_df is not None else 9

                    middle_offset = left_cols + 2

                    right_offset  = middle_offset + middle_cols + 2

                    pd.DataFrame([left_label]).to_excel(

                        writer, sheet_name=sheet_name, startrow=current_row, startcol=0, header=False, index=False

                    )

                    pd.DataFrame([middle_label]).to_excel(

                        writer, sheet_name=sheet_name, startrow=current_row, startcol=middle_offset, header=False, index=False

                    )

                    pd.DataFrame([right_label]).to_excel(

                        writer, sheet_name=sheet_name, startrow=current_row, startcol=right_offset, header=False, index=False

                    )

                    current_row += 1

                    if left_df is not None:

                        left_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, startcol=0, index=False)

                    if middle_df is not None:

                        middle_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, startcol=middle_offset, index=False)

                    if right_df is not None:

                        right_df.to_excel(writer, sheet_name=sheet_name, startrow=current_row, startcol=right_offset, index=False)

                    left_len   = len(left_df)   if left_df   is not None else 0

                    middle_len = len(middle_df) if middle_df is not None else 0

                    right_len  = len(right_df)  if right_df  is not None else 0

                    current_row += max(left_len, middle_len, right_len) + 4

            write_segment_list_to_sheet(overall_data, "Overall Deciles", "TRAIN", "VALIDATION", "OOS (TEST)")

            write_segment_list_to_sheet(cb_data, "Circlebacks", "TRAIN", "VALIDATION", "OOS (TEST)")

            write_segment_list_to_sheet(und_data, "Undecideds", "TRAIN", "VALIDATION", "OOS (TEST)")

            if age_overall_data:
                write_segment_list_to_sheet(age_overall_data, "Age Breakdown Overall", "TRAIN", "VALIDATION", "OOS (TEST)")

            if age_cb_data:
                write_segment_list_to_sheet(age_cb_data, "Age Breakdown Circlebacks", "TRAIN", "VALIDATION", "OOS (TEST)")

            if age_und_data:
                write_segment_list_to_sheet(age_und_data, "Age Breakdown Undecideds", "TRAIN", "VALIDATION", "OOS (TEST)")

            if 'Sheet' in writer.book.sheetnames:

                writer.book.remove(writer.book['Sheet'])

        print(f"✅ Detailed Excel Report generated locally: {filename}")

    except Exception as e:

        print(f"❌ Error generating Excel report: {e}")
 
# ==========================================

# 3. PDF GENERATOR CLASS

# ==========================================
 
class ClassificationPDFReport:

    def __init__(self, filename):

        self.filename = filename

        self.story = []

        self.styles = getSampleStyleSheet()

        self.width, self.height = letter

        self.styles['Title'].fontSize = 18

        self.styles['Title'].alignment = TA_CENTER

        self.styles['Heading1'].fontSize = 14

        self.styles['Heading1'].spaceAfter = 10

        self.styles['Heading2'].fontSize = 12

        self.intro_style = ParagraphStyle('IntroStyle', parent=self.styles['Normal'], fontSize=11, leading=15, spaceAfter=12, alignment=TA_JUSTIFY)

        self.feature_list_style = ParagraphStyle('FeatureListStyle', parent=self.styles['Normal'], fontSize=9, leading=12, alignment=TA_LEFT)

        self.table_text = ParagraphStyle('TableText', parent=self.styles['Normal'], fontSize=8, alignment=TA_CENTER)

        self.table_header = ParagraphStyle('TableHeader', parent=self.styles['Normal'], fontSize=8, textColor=colors.white, alignment=TA_CENTER, fontName='Helvetica-Bold')
 
    def add_heading(self, text):

        self.story.append(Paragraph(text, self.styles['Title']))

        self.story.append(Spacer(1, 0.2 * inch))
 
    def add_intro_text(self, text):

        self.story.append(Paragraph(text, self.intro_style))

        self.story.append(Spacer(1, 0.1 * inch))
 
    def add_feature_list(self, text):

        self.story.append(Paragraph(text, self.feature_list_style))

        self.story.append(Spacer(1, 0.15 * inch))
 
    def add_section_header(self, text):

        self.story.append(Paragraph(text, self.styles['Heading1']))

        self.story.append(Spacer(1, 0.1 * inch))
 
    def add_subheading(self, text):

        self.story.append(Paragraph(text, self.styles['Heading2']))

        self.story.append(Spacer(1, 0.1 * inch))
 
    def add_data_table(self, data, col_widths=None, header_color=colors.darkblue):

        formatted_data = []

        for i, row in enumerate(data):

            new_row = []

            for cell in row:

                style = self.table_header if i == 0 else self.table_text

                new_row.append(Paragraph(str(cell), style))

            formatted_data.append(new_row)

        if not col_widths:

            col_width = (self.width - 1.0*inch) / len(data[0])

            col_widths = [col_width] * len(data[0])

        t = Table(formatted_data, colWidths=col_widths)

        t.setStyle(TableStyle([

            ('BACKGROUND', (0, 0), (-1, 0), header_color),

            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),

            ('topPadding', (0,0), (-1,-1), 4),

            ('bottomPadding', (0,0), (-1,-1), 4),

        ]))

        self.story.append(t)

        self.story.append(Spacer(1, 0.25 * inch))
 
    def add_classification_report(self, report_dict):

        data = [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]

        for key, metrics in report_dict.items():

            if isinstance(metrics, dict):

                data.append([

                    str(key), 

                    f"{metrics.get('precision', 0):.4f}", 

                    f"{metrics.get('recall', 0):.4f}", 

                    f"{metrics.get('f1-score', 0):.4f}", 

                    str(int(metrics.get('support', 0)))

                ])

        self.add_data_table(data, col_widths=[1.5*inch, 1*inch, 1*inch, 1*inch, 1*inch], header_color=colors.slategrey)
 
    def add_confusion_matrix(self, cm):

        tn, fp, fn, tp = cm.ravel()

        data = [

            ['', 'Predicted Negative (0)', 'Predicted Positive (1)'],

            ['Actual Negative (0)', f"TN: {tn:,}", f"FP: {fp:,}"],

            ['Actual Positive (1)', f"FN: {fn:,}", f"TP: {tp:,}"]

        ]

        self.add_data_table(data, col_widths=[1.5*inch, 2.25*inch, 2.25*inch], header_color=colors.slategrey)
 
    def add_decile_table(self, df):

        if df is None or df.empty: return

        display_df = df.copy()

        for col in display_df.select_dtypes(include=['float']).columns:

            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

        data = [display_df.columns.tolist()]

        data.extend(display_df.values.tolist())

        self.add_data_table(data)

    def add_image(self, image_path, width=6.5*inch, height=4.5*inch):

        if os.path.exists(image_path):

            img = Image(image_path, width=width, height=height)

            self.story.append(img)

            self.story.append(Spacer(1, 0.25 * inch))
 
    def add_page_break(self):

        self.story.append(PageBreak())
 
    def build(self):

        doc = SimpleDocTemplate(self.filename, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)

        doc.build(self.story)
 
# ==========================================

# 4. CORE EVALUATION PIPELINE

# ==========================================
 
def evaluate_classification_models(train_df, val_df, oos_df, models_data, global_fallback_features, target_col, target_col_oos, segment_col, bucket_name):

    all_models_results = []

    local_results_dir = "Output/Results/"

    os.makedirs(local_results_dir, exist_ok=True)
 
    for item in models_data:

        model = item['model']

        features = item['features'] if item['features'] is not None else global_fallback_features

        tuning_log = item.get('tuning_log', None) 

        try:

            model_class_name = type(model).__name__

            if model_class_name == "Pipeline":

                model_class_name = type(model.named_steps['classifier']).__name__

            # 🚨 GUARANTEED CLEAN HEADERS FOR THE PDF 🚨

            display_name = model_class_name 

            model_params = str(model.get_params()) if hasattr(model, 'get_params') else str(model)

            if len(model_params) > 100: model_params = model_params[:100] + "..."

            model_name = f"{model_class_name} ({model_params})"

            model_name = model_name.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        except:

            display_name = "Unknown_Model"

            model_name = "Unknown_Model"

        print(f"\nEvaluating Model: {display_name}")
 
        X_train, y_train = train_df[features], train_df[target_col].astype(int)

        X_val, y_val = val_df[features], val_df[target_col].astype(int)

        X_oos, y_oos = oos_df[features], oos_df[target_col_oos].astype(int)
 
        train_probs = model.predict_proba(X_train)[:, 1]

        val_probs = model.predict_proba(X_val)[:, 1]

        oos_probs = model.predict_proba(X_oos)[:, 1]

        train_preds = model.predict(X_train)

        val_preds   = model.predict(X_val)

        oos_preds   = model.predict(X_oos)

        train_auc = roc_auc_score(y_train, train_probs)

        val_auc = roc_auc_score(y_val, val_probs)

        train_metrics = {

            'ROC_AUC':   train_auc,

            'Accuracy':  accuracy_score(y_train, train_preds),

            'Precision': precision_score(y_train, train_preds, zero_division=0),

            'Recall':    recall_score(y_train, train_preds, zero_division=0),

            'F1':        f1_score(y_train, train_preds, zero_division=0)

        }

        val_metrics = {

            'ROC_AUC':   val_auc,

            'Accuracy':  accuracy_score(y_val, val_preds),

            'Precision': precision_score(y_val, val_preds, zero_division=0),

            'Recall':    recall_score(y_val, val_preds, zero_division=0),

            'F1':        f1_score(y_val, val_preds, zero_division=0)

        }

        metrics = {

            'ROC_AUC': roc_auc_score(y_oos, oos_probs),

            'Accuracy': accuracy_score(y_oos, oos_preds),

            'Precision': precision_score(y_oos, oos_preds, zero_division=0),

            'Recall': recall_score(y_oos, oos_preds, zero_division=0),

            'F1': f1_score(y_oos, oos_preds, zero_division=0)

        }

        class_report = classification_report(y_oos, oos_preds, output_dict=True, zero_division=0)

        cm = confusion_matrix(y_oos, oos_preds)
 
        importance_df = pd.DataFrame()

        model_to_check = model.named_steps['classifier'] if type(model).__name__ == "Pipeline" else model

        if hasattr(model_to_check, 'feature_importances_'):

            importance_df = pd.DataFrame({'Feature': features, 'Importance': model_to_check.feature_importances_})

        elif hasattr(model_to_check, 'coef_'):

            importance_df = pd.DataFrame({'Feature': features, 'Importance': np.abs(model_to_check.coef_[0])})

        if not importance_df.empty:

            importance_df.sort_values(by='Importance', ascending=False, inplace=True)
 
        # ==========================================

        # 🚨 BULLETPROOF SHAP GENERATOR 🚨

        # ==========================================

        shap_plot_path = None

        try:

            print(f"   📊 Generating SHAP values for {display_name}...")

            plt.close('all') 

            X_shap_raw = X_oos.sample(n=min(500, len(X_oos)), random_state=42)

            if type(model).__name__ == "Pipeline" and 'scaler' in model.named_steps:

                X_shap_scaled = model.named_steps['scaler'].transform(X_shap_raw)

                X_shap_processed = pd.DataFrame(X_shap_scaled, columns=features)

                actual_model = model.named_steps['classifier']

            else:

                X_shap_processed = X_shap_raw.copy()

                actual_model = model

            model_type = type(actual_model).__name__
 
            # ---------------------------------------------------------

            # 🚨 NATIVE XGBOOST SHAP BYPASS

            # Bypasses the SHAP library's C++ parser entirely!

            # ---------------------------------------------------------

            if model_type == "XGBClassifier":

                import xgboost as xgb

                # Safely convert objects to categories for XGBoost DMatrix

                for col in X_shap_processed.columns:

                    if X_shap_processed[col].dtype == 'object':

                        X_shap_processed[col] = X_shap_processed[col].astype('category')

                # Ask XGBoost for SHAP values natively instead of using the shap library

                try:

                    dmat = xgb.DMatrix(X_shap_processed, enable_categorical=True)

                except:

                    dmat = xgb.DMatrix(X_shap_processed)

                contribs = actual_model.get_booster().predict(dmat, pred_contribs=True)

                # Slicing [:, :-1] removes the base_margin (bias) to isolate feature impacts

                shap_values_to_plot = contribs[:, :-1]
 
            # ---------------------------------------------------------

            # STANDARD SHAP ROUTE FOR ALL OTHER MODELS

            # ---------------------------------------------------------

            else:

                # Intelligently encode strings to integers to prevent C++ float errors

                for col in X_shap_processed.columns:

                    if X_shap_processed[col].dtype == 'object' or str(X_shap_processed[col].dtype) == 'category':

                        X_shap_processed[col] = X_shap_processed[col].astype('category').cat.codes

                tree_models = ["RandomForestClassifier", "ExtraTreesClassifier", "LGBMClassifier", "CatBoostClassifier", "DecisionTreeClassifier"]

                try:

                    if model_type in tree_models:

                        explainer = shap.TreeExplainer(actual_model)

                        shap_values = explainer.shap_values(X_shap_processed)

                    elif model_type == "LogisticRegression":

                        explainer = shap.LinearExplainer(actual_model, X_shap_processed)

                        shap_values = explainer.shap_values(X_shap_processed)

                    else:

                        background_data = shap.sample(X_shap_processed, 50) 

                        X_shap_small = X_shap_processed.iloc[:100] 

                        explainer = shap.KernelExplainer(actual_model.predict_proba, background_data)

                        shap_values = explainer.shap_values(X_shap_small)

                        X_shap_processed = X_shap_small

                except Exception as e_inner:

                    print(f"   [!] Primary Explainer failed ({e_inner}), routing to KernelExplainer...")

                    background_data = shap.sample(X_shap_processed, 50) 

                    X_shap_small = X_shap_processed.iloc[:100] 

                    explainer = shap.KernelExplainer(actual_model.predict_proba, background_data)

                    shap_values = explainer.shap_values(X_shap_small)

                    X_shap_processed = X_shap_small
 
                # Ensure 2D array format for plotting

                if isinstance(shap_values, list) and len(shap_values) > 1:

                    shap_values_to_plot = np.array(shap_values[1])

                else:

                    shap_values_to_plot = np.array(shap_values)

                if len(shap_values_to_plot.shape) == 3:

                    shap_values_to_plot = shap_values_to_plot[:, :, 1]
 
            # 🚨 DRAW THE PLOT (plot_type="bar" removed so it renders multi-color)

            shap.summary_plot(shap_values_to_plot, X_shap_processed, show=False)
 
            fig = plt.gcf()

            if fig.get_axes(): 

                safe_model_name = display_name.replace(" ", "_")[:20]

                shap_plot_path = os.path.join(local_results_dir, f"SHAP_{safe_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

                fig.savefig(shap_plot_path, dpi=150, bbox_inches='tight')

                print(f"   ✅ SHAP plot generated successfully.")

            plt.close('all')

        except Exception as e:

            print(f"   ⚠️ Could not generate SHAP for {display_name}. Skipping plot. Error: {e}")

            plt.close('all')

            shap_plot_path = None
 
        # --- DECILE CALCULATIONS (OVERALL) ---

        train_10 = create_quantile_table(train_probs, y_train, 10)

        train_20 = create_quantile_table(train_probs, y_train, 20)

        train_50 = create_quantile_table(train_probs, y_train, 50)

        train_100 = create_quantile_table(train_probs, y_train, 100)
 
        oos_data_mapping = pd.DataFrame({'PROB': oos_probs, 'TARGET': y_oos})

        oos_total_count = y_oos.sum()

        oos_decile_10 = create_valid_decile_analysis(get_cutoffs(train_10), 'PROB', 'TARGET', oos_data_mapping, oos_total_count)

        oos_decile_20 = create_valid_decile_analysis(get_cutoffs_ventile(train_20), 'PROB', 'TARGET', oos_data_mapping, oos_total_count)

        oos_decile_50 = create_valid_decile_analysis(get_cutoffs_generic(train_50), 'PROB', 'TARGET', oos_data_mapping, oos_total_count)

        oos_decile_100 = create_valid_decile_analysis(get_cutoffs_generic(train_100), 'PROB', 'TARGET', oos_data_mapping, oos_total_count)

        # Top 15% capture (top 3 of 20 buckets) for each dataset.
        # Train: directly from its own 20-split.
        # Val: build a Val mapping using TRAIN's cutoffs (same logic as OOS).
        # OOS: already computed below.

        train_top15_capture = train_20['Overall_Target PCT'].iloc[0:3].sum()

        val_data_mapping = pd.DataFrame({'PROB': val_probs, 'TARGET': y_val})

        val_total_count = y_val.sum()

        val_decile_10  = create_valid_decile_analysis(get_cutoffs(train_10), 'PROB', 'TARGET', val_data_mapping, val_total_count)

        val_decile_20  = create_valid_decile_analysis(get_cutoffs_ventile(train_20), 'PROB', 'TARGET', val_data_mapping, val_total_count)

        val_decile_50  = create_valid_decile_analysis(get_cutoffs_generic(train_50), 'PROB', 'TARGET', val_data_mapping, val_total_count)

        val_decile_100 = create_valid_decile_analysis(get_cutoffs_generic(train_100), 'PROB', 'TARGET', val_data_mapping, val_total_count)

        val_top15_capture = val_decile_20['Overall_Target PCT'].iloc[0:3].sum()

        top3_capture_rate = oos_decile_20['Overall_Target PCT'].iloc[0:3].sum()
 
        # --- SEGMENT DECILES ---

        cb_mask_train = train_df[segment_col] == 0

        und_mask_train = train_df[segment_col] == 1

        cb_mask_val = val_df[segment_col] == 0

        und_mask_val = val_df[segment_col] == 1

        cb_mask_oos = oos_df[segment_col] == 0

        und_mask_oos = oos_df[segment_col] == 1

        # All segment tables (Train/Val/OOS × CB/UND) use OVERALL Train cutoffs
        # so bucket boundaries are identical across Overall / CB / UND panels.

        overall_cutoffs_10 = get_cutoffs(train_10)
        overall_cutoffs_20 = get_cutoffs_ventile(train_20)

        # --- Train segment mappings ---
        train_cb_mapping = pd.DataFrame({'PROB': train_probs[cb_mask_train], 'TARGET': y_train[cb_mask_train]})
        train_cb_total   = y_train[cb_mask_train].sum()
        train_cb_10 = create_valid_decile_analysis(overall_cutoffs_10, 'PROB', 'TARGET', train_cb_mapping, train_cb_total)
        train_cb_20 = create_valid_decile_analysis(overall_cutoffs_20, 'PROB', 'TARGET', train_cb_mapping, train_cb_total)

        train_und_mapping = pd.DataFrame({'PROB': train_probs[und_mask_train], 'TARGET': y_train[und_mask_train]})
        train_und_total   = y_train[und_mask_train].sum()
        train_und_10 = create_valid_decile_analysis(overall_cutoffs_10, 'PROB', 'TARGET', train_und_mapping, train_und_total)
        train_und_20 = create_valid_decile_analysis(overall_cutoffs_20, 'PROB', 'TARGET', train_und_mapping, train_und_total)

        # --- Val segment mappings ---
        val_cb_mapping = pd.DataFrame({'PROB': val_probs[cb_mask_val], 'TARGET': y_val[cb_mask_val]})
        val_cb_total   = y_val[cb_mask_val].sum()
        val_cb_10 = create_valid_decile_analysis(overall_cutoffs_10, 'PROB', 'TARGET', val_cb_mapping, val_cb_total)
        val_cb_20 = create_valid_decile_analysis(overall_cutoffs_20, 'PROB', 'TARGET', val_cb_mapping, val_cb_total)

        val_und_mapping = pd.DataFrame({'PROB': val_probs[und_mask_val], 'TARGET': y_val[und_mask_val]})
        val_und_total   = y_val[und_mask_val].sum()
        val_und_10 = create_valid_decile_analysis(overall_cutoffs_10, 'PROB', 'TARGET', val_und_mapping, val_und_total)
        val_und_20 = create_valid_decile_analysis(overall_cutoffs_20, 'PROB', 'TARGET', val_und_mapping, val_und_total)

        # --- OOS segment mappings ---
        oos_cb_mapping = pd.DataFrame({'PROB': oos_probs[cb_mask_oos], 'TARGET': y_oos[cb_mask_oos]})
        oos_cb_total   = y_oos[cb_mask_oos].sum()
        oos_cb_10 = create_valid_decile_analysis(overall_cutoffs_10, 'PROB', 'TARGET', oos_cb_mapping, oos_cb_total)
        oos_cb_20 = create_valid_decile_analysis(overall_cutoffs_20, 'PROB', 'TARGET', oos_cb_mapping, oos_cb_total)

        oos_und_mapping = pd.DataFrame({'PROB': oos_probs[und_mask_oos], 'TARGET': y_oos[und_mask_oos]})
        oos_und_total   = y_oos[und_mask_oos].sum()
        oos_und_10 = create_valid_decile_analysis(overall_cutoffs_10, 'PROB', 'TARGET', oos_und_mapping, oos_und_total)
        oos_und_20 = create_valid_decile_analysis(overall_cutoffs_20, 'PROB', 'TARGET', oos_und_mapping, oos_und_total)

        # --- ACCOUNT AGE TABLES (OVERALL + SEGMENTS) ---

        print(f"\n   🔍 ACCOUNT_AGE diagnostic:")
        print(f"      train_df has ACCOUNT_AGE? {'ACCOUNT_AGE' in train_df.columns}")
        print(f"      val_df has ACCOUNT_AGE?   {'ACCOUNT_AGE' in val_df.columns}")
        print(f"      oos_df has ACCOUNT_AGE?   {'ACCOUNT_AGE' in oos_df.columns}")
        print(f"      train_df total cols: {len(train_df.columns)}")
        print(f"      Sample train_df cols: {list(train_df.columns)[:30]}")

        if 'ACCOUNT_AGE' in train_df.columns and \
           'ACCOUNT_AGE' in val_df.columns and \
           'ACCOUNT_AGE' in oos_df.columns:

            print(f"   ✅ Computing age tables...")

            try:
                # OVERALL — everyone together
                train_age = create_account_age_table(train_df, target_col, train_probs)
                val_age   = create_account_age_table(val_df, target_col, val_probs)
                oos_age   = create_account_age_table(oos_df, target_col_oos, oos_probs)
                age_overall_data = [
                    ("Account Age (7 Bins) - Overall", train_age, val_age, oos_age)
                ]
                print(f"      ✅ Overall age table built: {len(train_age)} rows")

                # CIRCLEBACKS (FIRSTACTION_BOOL == 0)
                train_age_cb = create_account_age_table(
                    train_df[cb_mask_train].reset_index(drop=True), target_col, train_probs[cb_mask_train.values]
                )
                val_age_cb = create_account_age_table(
                    val_df[cb_mask_val].reset_index(drop=True), target_col, val_probs[cb_mask_val.values]
                )
                oos_age_cb = create_account_age_table(
                    oos_df[cb_mask_oos].reset_index(drop=True), target_col_oos, oos_probs[cb_mask_oos.values]
                )
                age_cb_data = [
                    ("Account Age (7 Bins) - Circlebacks", train_age_cb, val_age_cb, oos_age_cb)
                ]
                print(f"      ✅ Circleback age table built")

                # UNDECIDEDS (FIRSTACTION_BOOL == 1)
                train_age_und = create_account_age_table(
                    train_df[und_mask_train].reset_index(drop=True), target_col, train_probs[und_mask_train.values]
                )
                val_age_und = create_account_age_table(
                    val_df[und_mask_val].reset_index(drop=True), target_col, val_probs[und_mask_val.values]
                )
                oos_age_und = create_account_age_table(
                    oos_df[und_mask_oos].reset_index(drop=True), target_col_oos, oos_probs[und_mask_oos.values]
                )
                age_und_data = [
                    ("Account Age (7 Bins) - Undecideds", train_age_und, val_age_und, oos_age_und)
                ]
                print(f"      ✅ Undecided age table built")
                print(f"      Final: age_overall_data has {len(age_overall_data)} entries")
            except Exception as age_err:
                print(f"      ❌ Age table error: {age_err}")
                age_overall_data = []
                age_cb_data = []
                age_und_data = []
        else:
            age_overall_data = []
            age_cb_data = []
            age_und_data = []
            print("   ⚠️ ACCOUNT_AGE not found in data — skipping age breakdown")

        # ==========================================

        # GENERATE GRANULAR EXCEL & PUSH TO S3

        # ==========================================

        safe_model_name = display_name.replace(" ", "_")[:20]

        excel_filename = os.path.join(local_results_dir, f"Granular_Deciles_{safe_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

        overall_excel_data = [

            ("10 Split",  train_10,  val_decile_10,  oos_decile_10),

            ("20 Split",  train_20,  val_decile_20,  oos_decile_20),

            ("50 Split",  train_50,  val_decile_50,  oos_decile_50),

            ("100 Split", train_100, val_decile_100, oos_decile_100)

        ]

        cb_excel_data = [

            (f"Circlebacks ({segment_col}=0) - 10 Split", train_cb_10, val_cb_10, oos_cb_10),

            (f"Circlebacks ({segment_col}=0) - 20 Split", train_cb_20, val_cb_20, oos_cb_20)

        ]

        und_excel_data = [

            (f"Undecideds ({segment_col}=1) - 10 Split", train_und_10, val_und_10, oos_und_10),

            (f"Undecideds ({segment_col}=1) - 20 Split", train_und_20, val_und_20, oos_und_20)

        ]

        export_granular_excel(excel_filename, overall_excel_data, cb_excel_data, und_excel_data,
                              age_overall_data=age_overall_data,
                              age_cb_data=age_cb_data,
                              age_und_data=age_und_data)

        upload_file_to_s3(excel_filename, bucket_name, "Retraining_Framework_New/Output/Results/")
 
        # Pack PDF results

        all_models_results.append({

            'model_name': model_name,

            'display_name': display_name,   

            'train_auc': train_auc,

            'val_auc': val_auc,

            'train_metrics': train_metrics,

            'val_metrics': val_metrics,

            'metrics': metrics,

            'class_report': class_report,

            'confusion_matrix': cm,          

            'feature_importance': importance_df,

            'shap_plot_path': shap_plot_path,

            'tuning_log': tuning_log,

            'top3_capture_rate': top3_capture_rate,

            'train_top15_capture': train_top15_capture,

            'val_top15_capture':   val_top15_capture,

            'oos_decile_20': oos_decile_20

        })
 
    # ==========================================

    # SORT, GENERATE PDF, & PUSH TO S3

    # ==========================================

    all_models_results.sort(key=lambda x: (x['top3_capture_rate'], x['metrics']['ROC_AUC']), reverse=True)
 
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_IST")

    pdf_filename = os.path.join(local_results_dir, f"Model_Metrics_{timestamp_str}.pdf")

    pdf = ClassificationPDFReport(pdf_filename)

    # --- PAGE 1: INTRO & FEATURES ---

    pdf.add_heading("Model Performance & Classification Report")

    intro_text = f"""
<b>Guide to This Model Evaluation Report</b><br/><br/>

    This document contains a complete evaluation of the machine learning models trained to predict <b>{target_col}</b>. It provides all the necessary metrics, charts, and tables to understand how each model performs on unseen test data (Out-of-Sample).<br/><br/>
<b>What you will find in this PDF:</b><br/>
&bull; <b>Model Leaderboard:</b> A quick summary ranking all trained models based on their overall AUC and their ability to capture positive targets in the top tiers.<br/>
&bull; <b>Detailed Metrics & Confusion Matrix:</b> Standard evaluation scores along with the exact physical counts of correct and incorrect predictions.<br/>
&bull; <b>Hyperparameter Tuning History:</b> A log of the top combinations tested during the random search phase, showing how the engine hunted for the optimal configuration.<br/>
&bull; <b>Feature Importance & SHAP Plots:</b> Tables and visual charts explaining exactly which input features the model relies on the most, and how those features push the final prediction up or down.<br/>
&bull; <b>Out-of-Sample Decile Tables:</b> A 20-tier breakdown showing the model's performance grouped by probability scores.<br/><br/>
<i>Note: For the full 10, 50, and 100-split decile tables, please refer to the accompanying Excel files.</i>

    """

    pdf.add_intro_text(intro_text)

    pdf.add_section_header("Model Inputs (Selected Features)")

    features_str = ", ".join(str(f) for f in global_fallback_features)

    pdf.add_feature_list(features_str)
 
    # --- PAGE 2: LEADERBOARD ---

    pdf.add_page_break()

    pdf.add_section_header("Executive Summary: Model Leaderboard")

    # Full metrics matrix: for each model, show AUC / Accuracy / Precision / Recall / F1
    # for Train, Val, and OOS — plus the top-15% capture rate.

    leaderboard_data = [[

        'Rank', 'Model Name', 'Dataset',

        'AUC', 'Accuracy', 'Precision', 'Recall', 'F1',

        'Top 15% Capture'

    ]]

    for idx, res in enumerate(all_models_results):

        rank_str  = str(idx + 1)

        name_str  = res['display_name']

        # Top 15% capture for each dataset

        capture_by_dataset = {

            'Train': f"{res.get('train_top15_capture', 0):.2f}%",

            'Val':   f"{res.get('val_top15_capture',   0):.2f}%",

            'OOS':   f"{res.get('top3_capture_rate',   0):.2f}%",

        }

        for dataset_label, dataset_metrics in [

            ('Train', res.get('train_metrics', {})),

            ('Val',   res.get('val_metrics',   {})),

            ('OOS',   res.get('metrics',       {})),

        ]:

            leaderboard_data.append([

                rank_str if dataset_label == 'Train' else '',

                name_str if dataset_label == 'Train' else '',

                dataset_label,

                f"{dataset_metrics.get('ROC_AUC', 0):.4f}",

                f"{dataset_metrics.get('Accuracy', 0):.4f}",

                f"{dataset_metrics.get('Precision', 0):.4f}",

                f"{dataset_metrics.get('Recall', 0):.4f}",

                f"{dataset_metrics.get('F1', 0):.4f}",

                capture_by_dataset[dataset_label],

            ])

    pdf.add_data_table(leaderboard_data)

    # --- MODEL DETAILS LOOP ---

    for idx, res in enumerate(all_models_results):

        # --- SUB-PAGE 1: METRICS, CM & TUNING LOG ---

        pdf.add_page_break()

        pdf.add_section_header(f"Detailed Analysis: {res['display_name']}")

        pdf.add_subheading("Overall Test (OOS) Metrics")

        metrics_data = [['Metric', 'Score']]

        for k, v in res['metrics'].items():

            metrics_data.append([k, f"{v:.4f}"])

        pdf.add_data_table(metrics_data, col_widths=[2*inch, 2*inch])
 
        pdf.add_subheading("Classification Report (Test Data)")

        pdf.add_classification_report(res['class_report'])
 
        pdf.add_subheading("Confusion Matrix (Test Data)")

        pdf.add_confusion_matrix(res['confusion_matrix'])

        if res.get('tuning_log') is not None and not res['tuning_log'].empty:

            pdf.add_subheading("Hyperparameter Tuning History (Top 5 Configurations)")

            log_df = res['tuning_log'].head(5)

            log_data = [['Rank', 'CV PR-AUC', 'Train Time', 'Tested Hyperparameters']]

            for _, row in log_df.iterrows():

                params_str = str(row['params']).replace("{", "").replace("}", "").replace("'", "")

                log_data.append([

                    str(row['Rank']),

                    f"{row['PR_AUC_Score']:.4f}",

                    f"{row['Train_Time_s']:.1f}s",

                    params_str

                ])

            pdf.add_data_table(log_data, col_widths=[0.6*inch, 1*inch, 1*inch, 4.4*inch], header_color=colors.cadetblue)

        # --- SUB-PAGE 2: TABULAR FEATURE IMPORTANCE ---

        pdf.add_page_break()

        pdf.add_section_header(f"Feature Importances: {res['display_name']}")

        if not res['feature_importance'].empty:

            fi_data = [['Feature Name', 'Importance']]

            fi_formatted = [[row['Feature'], f"{row['Importance']:.4f}"] for _, row in res['feature_importance'].iterrows()]

            fi_data.extend(fi_formatted)

            pdf.add_data_table(fi_data, col_widths=[3*inch, 2*inch])

        else:

            pdf.story.append(Paragraph("Feature importance not supported for this model type.", pdf.styles['Normal']))

            pdf.story.append(Spacer(1, 0.2 * inch))
 
        # --- SUB-PAGE 3: SHAP PLOT ---

        if res.get('shap_plot_path') and os.path.exists(res['shap_plot_path']):

            pdf.add_page_break()

            pdf.add_section_header(f"SHAP Feature Explanations: {res['display_name']}")

            pdf.add_image(res['shap_plot_path'])
 
        # --- SUB-PAGE 4: DECILE TABLE ---

        pdf.add_page_break()

        pdf.add_section_header(f"Out-of-Sample Mapping (20-Split): {res['display_name']}")

        pdf.add_decile_table(res['oos_decile_20'])
 
    pdf.build()

    print(f"✅ Success! Main PDF Report generated locally at {pdf_filename}")

    upload_file_to_s3(pdf_filename, bucket_name, "Retraining_Framework_New/Output/Results/")
 
    # Clean up SHAP image files from the local disk

    for res in all_models_results:

        if res.get('shap_plot_path') and os.path.exists(res['shap_plot_path']):

            try:

                os.remove(res['shap_plot_path'])

            except:

                pass
 
# ==========================================

# 5. S3 HELPER FUNCTIONS

# ==========================================
 
def upload_file_to_s3(local_file_path, bucket_name, s3_folder):

    file_name = os.path.basename(local_file_path)

    s3_key = f"{s3_folder}{file_name}"

    print(f"📤 Uploading {file_name} to s3://{bucket_name}/{s3_key}")

    s3_client.upload_file(local_file_path, bucket_name, s3_key)
 
def get_latest_file_from_s3(bucket_name: str, folder_prefix: str) -> str:

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix)

    if "Contents" not in response:

        raise ValueError(f"No files found in {folder_prefix}")

    files = [obj for obj in response["Contents"] if not obj["Key"].endswith("/")]

    if not files:

        raise ValueError(f"No files found in {folder_prefix}")

    latest_file = max(files, key=lambda x: x["LastModified"])

    return latest_file["Key"]
 
def read_csv_from_s3(bucket_name: str, file_key: str) -> pd.DataFrame:

    print(f"📥 Fetching CSV from S3: s3://{bucket_name}/{file_key}")

    obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)

    return pd.read_csv(BytesIO(obj["Body"].read()))
 
def get_latest_model_keys(bucket: str, models_prefix: str, project_name: str, selected_models: list) -> list:

    prefix = f"{models_prefix}{project_name}__"

    paginator = s3_client.get_paginator("list_objects_v2")

    page_iter = paginator.paginate(Bucket=bucket, Prefix=prefix)

    latest_by_model = {}

    for page in page_iter:

        if "Contents" not in page: continue

        for obj in page["Contents"]:

            key = obj["Key"]

            if not key.endswith(".pkl") or key.endswith("/"): continue

            filename = os.path.basename(key)

            parts = filename.split("__")

            if len(parts) != 3: continue

            proj, model_name, ts_part = parts

            if proj != project_name or model_name not in selected_models:

                continue

            timestamp_str = ts_part.replace(".pkl", "")

            if model_name not in latest_by_model or timestamp_str > latest_by_model[model_name][0]:

                latest_by_model[model_name] = (timestamp_str, key)

    return [info[1] for info in latest_by_model.values()]
 
def load_models_and_features(bucket, model_s3_keys):

    models_data = []

    for key in model_s3_keys:

        response = s3_client.get_object(Bucket=bucket, Key=key)

        loaded_obj = pickle.load(io.BytesIO(response['Body'].read()))

        if isinstance(loaded_obj, dict) and 'model' in loaded_obj:

            models_data.append({

                'model': loaded_obj['model'],

                'features': loaded_obj.get('features'),

                'tuning_log': loaded_obj.get('tuning_log', None)

            })

        else:

            models_data.append({

                'model': loaded_obj,

                'features': getattr(loaded_obj, 'feature_names_in_', None),

                'tuning_log': None

            })

    return models_data
 
def load_metadata_from_s3(bucket_name: str, s3_key: str) -> dict:

    print(f"📥 Fetching Metadata from S3: s3://{bucket_name}/{s3_key}")

    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)

    return json.loads(response['Body'].read().decode('utf-8'))
 
# ==========================================

# 6. MAIN EXECUTION

# ==========================================

if __name__ == "__main__": 

    catalog = load_model_catalog()
    BUCKET_NAME, _folders = get_s3_config(catalog)
    MODELS_PREFIX = _folders["model_output"]

    project_name = list(catalog.get("production_models", {}).keys())[0]

    selection_key = get_latest_file_from_s3(BUCKET_NAME, f"{MODELS_PREFIX}{project_name}__selection_metadata__")

    metadata = load_metadata_from_s3(BUCKET_NAME, selection_key)

    extracted_features = metadata.get("selected_features", [])

    selected_models = metadata.get("selected_models", [])

    if not extracted_features or not selected_models:

        raise ValueError("Critical metadata (features or models list) missing. Cannot proceed.")
 
    print(f"🔍 Filtering S3 for exactly these models: {selected_models}")

    MODEL_KEYS = get_latest_model_keys(bucket=BUCKET_NAME, models_prefix=MODELS_PREFIX, project_name=project_name, selected_models=selected_models)

    loaded_models_data = load_models_and_features(BUCKET_NAME, MODEL_KEYS)
 
    DATA_FOLDER = _folders["preprocessed_train"].rstrip("/")

    VAL_FOLDER  = _folders["preprocessed_val"].rstrip("/")

    OOS_FOLDER  = _folders["preprocessed_oos"].rstrip("/")

    train_key = get_latest_file_from_s3(BUCKET_NAME, f"{DATA_FOLDER}/train_data_new_")

    val_key   = get_latest_file_from_s3(BUCKET_NAME, f"{VAL_FOLDER}/val_data_new_")

    oos_key   = get_latest_file_from_s3(BUCKET_NAME, f"{OOS_FOLDER}/oos_data_new_")
 
    train_df = read_csv_from_s3(BUCKET_NAME, train_key)

    val_df   = read_csv_from_s3(BUCKET_NAME, val_key)

    oos_df   = read_csv_from_s3(BUCKET_NAME, oos_key)

    if train_df is not None and loaded_models_data:

        evaluate_classification_models(

            train_df=train_df, 

            val_df=val_df, 

            oos_df=oos_df, 

            models_data=loaded_models_data, 

            global_fallback_features=extracted_features,

            target_col=TARGET_COLUMN,    

            target_col_oos=TARGET_COLUMN_OOS,

            segment_col=SEGMENT_COLUMN,      

            bucket_name=BUCKET_NAME          

        )

    else:

        print("Error: Missing data or models.")
 