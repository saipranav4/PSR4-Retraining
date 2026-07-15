from reportlab.platypus import (

    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, ListFlowable, ListItem

)

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from reportlab.lib.pagesizes import letter

from reportlab.lib import colors

from reportlab.lib.units import inch

from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

import matplotlib.pyplot as plt

import numpy as np

from datetime import datetime

import io

import contextlib

import pandas as pd
 
from data_check import (

    schema_check, data_type_check, duplicate_check, missing_value_check,

    long_tail_test, length_distribution_with_long_tail, outlier_check,

    unique_value_check, data_distribution_check

)
 
def int_to_roman(n):

    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]

    syb = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]

    roman_num = ''

    i = 0

    while n > 0:

        for _ in range(n // val[i]):

            roman_num += syb[i]

            n -= val[i]

        i += 1

    return roman_num
 
def add_page_number(canvas, doc):

    page_num = canvas.getPageNumber()

    text = "Page %s" % page_num

    canvas.drawRightString(letter[0] - inch, 0.75 * inch, text)
 
class DataValidationReporter:

    def __init__(self, train_df, oos_df,

                 not_considered_columns=None, cat_cols_fixed=None,

                 numeric_cols_outlier=None, long_tail_cols=None,

                 text_cols_length_distribution=None, duplicate_subset_cols=None,

                 alt_model_features=None, train_date_from=None, train_date_to=None,

                 oos_date_from=None, oos_date_to=None):
 
        self.train_df = train_df

        self.oos_df = oos_df

        self.results = {}

        self.inch = inch
 
        self.not_considered_columns = not_considered_columns or []

        self.cat_cols_fixed = cat_cols_fixed or []

        self.numeric_cols_outlier = numeric_cols_outlier or []

        self.long_tail_cols = long_tail_cols or []

        self.text_cols_length_distribution = text_cols_length_distribution or []

        self.duplicate_subset_cols = duplicate_subset_cols

        self.alt_model_features = alt_model_features or []
 
        self.train_from = train_date_from

        self.train_to = train_date_to

        self.oos_from = oos_date_from

        self.oos_to = oos_date_to
 
        # --- STYLES ---

        self.styles = getSampleStyleSheet()

        self.styles.add(ParagraphStyle(name='ReportTitle', fontSize=22, leading=26, alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica-Bold'))

        self.styles.add(ParagraphStyle(name='ExecutiveSummaryTitle', fontSize=24, leading=28, spaceAfter=20, fontName='Helvetica-Bold', alignment=TA_CENTER))

        self.styles.add(ParagraphStyle(name='ReportHeading1', fontSize=18, leading=22, fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=15, textColor=colors.darkblue))

        self.styles.add(ParagraphStyle(name='ReportHeading2', fontSize=16, leading=20, fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=10))

        self.styles.add(ParagraphStyle(name='IntroText', fontSize=11, leading=16, spaceAfter=8, alignment=TA_JUSTIFY))

        self.styles.add(ParagraphStyle(name='Description', fontSize=10, leading=14, spaceAfter=4, textColor=colors.darkslategray))

        self.styles.add(ParagraphStyle(name='FinalResultFail', fontSize=14, leading=18, fontName='Helvetica-Bold', textColor=colors.red))

        self.styles.add(ParagraphStyle(name='FinalResultWarning', fontSize=14, leading=18, fontName='Helvetica-Bold', textColor=colors.darkorange))

        self.styles.add(ParagraphStyle(name='FinalResultPass', fontSize=14, leading=18, fontName='Helvetica-Bold', textColor=colors.green))

        self.styles.add(ParagraphStyle(name='SummaryPassDetail', fontSize=11, leading=14, textColor=colors.green, fontName='Helvetica-Bold'))

        self.styles.add(ParagraphStyle(name='SummaryWarningDetail', fontSize=11, leading=14, textColor=colors.darkorange, fontName='Helvetica-Bold'))

        self.styles.add(ParagraphStyle(name='Log', fontSize=9, leading=12, fontName='Courier', leftIndent=10))

        self.styles.add(ParagraphStyle(name='LogFail', fontSize=9, leading=12, fontName='Courier-Bold', leftIndent=10, textColor=colors.red))
 
    def get_status_style(self, status_text):

        if 'FAIL' in status_text: return self.styles['FinalResultFail']

        if 'WARNING' in status_text: return self.styles['FinalResultWarning']

        return self.styles['FinalResultPass']
 
    def run_all_tests(self):

        print("Running Schema Check...")

        self.results['Schema Check'] = self._schema_check()

        print("Running Data Type Check...")

        self.results['Data Type Check'] = self._data_type_check()

        print("Running Duplicate Check...")

        self.results['Duplicate Check'] = self._duplicate_check()

        print("Running Missing Value Check...")

        self.results['Missing Value Check'] = self._missing_value_check()

        print("Running Long Tail Check...")

        self.results['Long Tail Check'] = self._long_tail_test()

        print("Running Length Distribution Check...")

        self.results['Length Distribution Check'] = self._length_distribution_check()

        print("Running Outlier Check...")

        self.results['Outlier Check'] = self._outlier_check()

        print("Running Unique Value Check...")

        self.results['Unique Value Check'] = self._unique_value_check()

        print("Running Data Distribution Check...")

        self.results['Data Distribution Check'] = self._data_distribution_check()

        print("Compiling final data comparison result...")

        self.results['Data Comparison Result'] = self._data_comparison_result()

        print("\nAll tests complete.")
 
    def _data_comparison_result(self):

        final_status = "PASS"

        flowables = []
 
        schema_stat = self.results.get('Schema Check', {}).get('status', 'PASS')

        dtype_stat = self.results.get('Data Type Check', {}).get('status', 'PASS')

        if schema_stat == 'FAIL' or dtype_stat == 'FAIL':

            final_status = "FAIL"

            flowables.append(Paragraph(f"Overall Status: {final_status}", self.get_status_style(final_status)))

            flowables.append(Paragraph("A critical structural or data type mismatch exists between Train and OOS data. Models cannot be reliably trained.", self.styles['Normal']))

            return {'status': final_status, 'details_flowables': flowables}
 
        warning_tests = ['Missing Value Check', 'Outlier Check', 'Data Distribution Check', 'Unique Value Check', 'Long Tail Check']

        failed_warnings = [t for t in warning_tests if self.results.get(t, {}).get('status', 'PASS') == 'FAIL']

        if failed_warnings or self.results.get('Duplicate Check', {}).get('status') == 'FAIL':

            final_status = "PASS with WARNING"

            flowables.append(Paragraph(f"Overall Status: {final_status}", self.get_status_style(final_status)))

            flowables.append(Paragraph("Structural checks passed, but distributional drift was detected between Train and OOS data.", self.styles['Normal']))

        else:

            flowables.append(Paragraph(f"Overall Status: {final_status}", self.get_status_style(final_status)))

            flowables.append(Paragraph("Train and OOS datasets are perfectly consistent.", self.styles['Normal']))
 
        return {'status': final_status, 'details_flowables': flowables}
 
    def generate_report(self, filename="data_validation_report.pdf"):

        doc = SimpleDocTemplate(filename, pagesize=letter)

        story = []
 
        # --- TITLE PAGE & INTRODUCTION ---

        story.append(Paragraph("Data Consistency & Drift Report", self.styles['ReportTitle']))

        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal']))

        story.append(Spacer(1, 0.25*inch))
 
        # Dataset Table

        table_data = [

            ["Dataset", "Records", "Date From", "Date To"],

            ["Training Data", f"{self.train_df.shape[0]:,}", self.train_from, self.train_to],

            ["Out-of-Sample (OOS)", f"{self.oos_df.shape[0]:,}", self.oos_from, self.oos_to]

        ]

        data_table = Table(table_data, colWidths=[2.2*inch, 1*inch, 1.5*inch, 1.5*inch])

        data_table.setStyle(TableStyle([

            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),

            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),

            ('GRID', (0, 0), (-1, -1), 1, colors.black),

            ('ALIGN', (1,0), (-1,-1), 'CENTER'),

        ]))

        story.append(data_table)

        story.append(Spacer(1, 0.3*inch))
 
        # 🚨 THE 10-LINE INTRODUCTION 🚨

        story.append(Paragraph("1. INTRODUCTION TO THIS REPORT", self.styles['ReportHeading1']))

        intro_text = [

            "This document evaluates the structural and statistical alignment between your <b>Training dataset</b> and your <b>Out-of-Sample (OOS) dataset</b>.",

            "Its primary purpose is to guarantee that the machine learning pipeline will be scored on real-world data that identically matches the baseline data it was trained on.",

            "When training data and OOS data diverge (a phenomenon known as Data Drift), models can fail silently, produce highly inaccurate predictions, or crash entirely during inference.",

            "To prevent these critical failures, this report automatically sweeps your datasets and checks the following dimensions of data quality:"

        ]

        for p in intro_text:

            story.append(Paragraph(p, self.styles['IntroText']))

        intro_bullets = [

            "<b>Structural Integrity:</b> Ensuring columns, data types, and row uniqueness remain identical.",

            "<b>Missingness Stability:</b> Verifying that the rate of Null/NaN values has not significantly shifted.",

            "<b>Distributional Consistency:</b> Checking numerical central tendencies (mean/median) and outlier bounds.",

            "<b>Categorical Stability:</b> Ensuring high and low cardinality text features maintain consistent population proportions."

        ]

        list_items = [ListItem(Paragraph(b, self.styles['IntroText'])) for b in intro_bullets]

        story.append(ListFlowable(list_items, bulletType='bullet', leftIndent=15))

        story.append(Spacer(1, 0.1*inch))

        story.append(Paragraph("By reviewing this document, data scientists and engineers can quickly identify if the new data is safe for model inference or if upstream data pipelines require immediate remediation.", self.styles['IntroText']))
 
        story.append(PageBreak())
 
        # --- EXECUTIVE SUMMARY ---

        story.append(Paragraph("2. EXECUTIVE SUMMARY", self.styles['ExecutiveSummaryTitle']))

        defs_text = """<b>Status Meanings:</b><br/>
<font color='red'><b>FAIL:</b></font> A critical structural mismatch. Model training will likely crash.<br/>
<font color='darkorange'><b>WARNING:</b></font> Distributions or missing values shifted between Train and OOS. Investigate for Data Drift.<br/>
<font color='green'><b>PASS:</b></font> Data is perfectly consistent. """

        story.append(Paragraph(defs_text, self.styles['Normal']))

        story.append(Spacer(1, 0.2*inch))
 
        for test_name, result in self.results.items():

            if test_name == 'Data Comparison Result': continue

            status = result.get('status', 'FAIL')

            if status == 'FAIL' and test_name not in ['Schema Check', 'Data Type Check']: status = 'WARNING'

            style = self.get_status_style(status)

            story.append(Paragraph(f"<b>{test_name}:</b> {status}", style))

            story.append(Spacer(1, 0.1*inch))
 
        # 🚨 THE BULLETED TEST DESCRIPTIONS 🚨

        test_descriptions = {

            'Schema Check': [

                "Verifies that the Training and Out-of-Sample datasets share the exact same structural schema.",

                "Flags any missing or extra columns between the two datasets.",

                "Ensures the model receives the exact same features during scoring as it did during training."

            ],

            'Data Type Check': [

                "Ensures that all common columns have identical data types (e.g., float64, int, object).",

                "Prevents models from crashing during inference due to type mismatches."

            ],

            'Duplicate Check': [

                "Scans both datasets for duplicated rows based on specified subset columns.",

                "Excessive duplicates in training data can cause models to severely overfit."

            ],

            'Missing Value Check': [

                "Compares the proportion of Null/NaN values for each feature between the datasets.",

                "Flags features where the missingness rate shifts significantly (e.g., >10% delta).",

                "Helps identify broken data pipelines or changes in upstream data collection behavior."

            ],

            'Outlier Check': [

                "Evaluates numerical features to ensure the proportion of extreme values remains consistent.",

                "Uses Interquartile Range (IQR) bounds established on the Training data to evaluate the OOS data.",

                "Prevents shifts in outlier distributions that could degrade distance-based models."

            ],

            'Unique Value Check': [

                "Validates low-cardinality categorical features (like STATE or SEGMENT).",

                "Ensures the OOS data doesn't introduce completely unseen categories to the model.",

                "Checks that the relative distribution of these categories remains mathematically stable."

            ],

            'Data Distribution Check': [

                "Compares the central tendencies (Mean and Median) of all numerical features.",

                "A large percentage shift in the mean or median indicates severe data drift.",

                "Flags features where the OOS data is fundamentally different than the Training data."

            ],

            'Long Tail Check': [

                "Analyzes high-cardinality categorical features (like TAXONOMY or IDs).",

                "Ensures both frequent (head) and infrequent (tail) categories are distributed similarly.",

                "Prevents the model from being surprised by rare categories suddenly becoming common."

            ],

            'Length Distribution Check': [

                "Monitors the character length of text and identifier columns.",

                "Detects upstream data truncation, formatting changes, or joining errors."

            ]

        }
 
        # --- DETAILED REPORT (WITH PAGE BREAKS) ---

        story.append(PageBreak())

        story.append(Paragraph("3. DETAILED SUMMARY", self.styles['ExecutiveSummaryTitle']))

        story.append(Paragraph("This section contains the in-depth logs, dataframes, and visual plots for each validation check.", self.styles['IntroText']))
 
        for test_name, result in self.results.items():

            if test_name == 'Data Comparison Result': continue

            # 🚨 GUARANTEED PAGE BREAK FOR EVERY SECTION 🚨

            story.append(PageBreak())

            story.append(Paragraph(f"{test_name}", self.styles['ReportHeading1']))

            # Insert Bulleted Description

            desc_items = test_descriptions.get(test_name, ["No description available."])

            list_items = [ListItem(Paragraph(desc, self.styles['Description'])) for desc in desc_items]

            story.append(ListFlowable(list_items, bulletType='bullet', leftIndent=15))

            story.append(Spacer(1, 0.2*inch))

            # Overall Status logic

            status = result.get('status', 'FAIL')

            if status == 'FAIL' and test_name not in ['Schema Check', 'Data Type Check']: status = 'WARNING'

            story.append(Paragraph(f"<b>Overall Status:</b> {status}", self.get_status_style(status)))

            story.append(Spacer(1, 0.15*inch))

            # 🚨 CLEAR PASS / FAIL INDICATION 🚨

            if status == 'PASS':

                story.append(Paragraph("✅ <b>All checks passed perfectly.</b> No data drift, missing value shifts, or structural mismatches were detected.", self.styles['SummaryPassDetail']))

                story.append(Spacer(1, 0.15*inch))

            else:

                story.append(Paragraph("⚠️ <b>Discrepancies Detected:</b> Review the logs and plots below to identify specific feature failures.", self.styles['SummaryWarningDetail']))

                story.append(Spacer(1, 0.15*inch))

            # Logs

            if result.get('logs'):

                story.append(Paragraph("<b>Detailed Logs:</b>", self.styles['ReportHeading2']))

                for line in result['logs'].split('\n'):

                    if line.strip(): 

                        if 'FAILED' in line or 'Shift' in line or 'Mismatch' in line:

                            story.append(Paragraph(line, self.styles['LogFail']))

                        else:

                            story.append(Paragraph(line, self.styles['Log']))

            # Dataframes

            if 'dataframe_results' in result:

                for title, df in result['dataframe_results'].items():

                    if not df.empty:

                        df_str = df.round(4).astype(str)

                        tbl = Table([df_str.columns.tolist()] + df_str.values.tolist(), repeatRows=1)

                        tbl.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('FONTSIZE', (0, 0), (-1, -1), 6)]))

                        story.append(Spacer(1, 0.15*inch))

                        story.append(tbl)
 
            # Plots

            if 'plots' in result and result['plots']:

                story.append(Spacer(1, 0.15*inch))

                for img_buffer in result['plots']:

                    img_buffer.seek(0)

                    story.append(Image(img_buffer, width=6.5*inch, height=4*inch))

                    story.append(Spacer(1, 0.1*inch))
 
        doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

        print(f"✅ Report successfully generated: {filename}")
 
    # --- Test Wrapper Functions ---

    def _run_test(self, test_func, *args, **kwargs):

        plt.close('all')

        with io.StringIO() as buf, contextlib.redirect_stdout(buf):

            result = test_func(*args, **kwargs)

            logs = buf.getvalue()

        return result, logs
 
    def _schema_check(self):

        result, logs = self._run_test(schema_check, self.train_df, self.oos_df)

        passed, fails, _, _ = result

        return {'status': "PASS" if passed else "FAIL", 'logs': logs + f"\nFails: {fails}"}
 
    def _data_type_check(self):

        result, logs = self._run_test(data_type_check, self.train_df, self.oos_df)

        return {'status': "PASS" if result[0] else "FAIL", 'logs': logs}
 
    def _duplicate_check(self):

        result, logs = self._run_test(duplicate_check, self.train_df, self.oos_df, self.duplicate_subset_cols)

        return {'status': "PASS" if result[0] else "FAIL", 'logs': logs + result[1]}
 
    def _missing_value_check(self):

        result, logs = self._run_test(missing_value_check, self.train_df, self.oos_df, self.not_considered_columns)

        return {'status': "PASS" if result[0] else "FAIL", 'logs': logs, 'plots': result[2]}
 
    def _long_tail_test(self):

        result, logs = self._run_test(long_tail_test, self.train_df, self.oos_df, self.long_tail_cols)

        return {'status': "PASS" if result[0] else "FAIL", 'logs': logs, 'plots': result[2]}
 
    def _length_distribution_check(self):

        result, logs = self._run_test(length_distribution_with_long_tail, self.train_df, self.oos_df, self.text_cols_length_distribution)

        return {'status': "PASS" if result[0] else "FAIL", 'logs': logs, 'plots': result[2]}
 
    def _outlier_check(self):

        result, logs = self._run_test(outlier_check, self.train_df, self.oos_df, self.numeric_cols_outlier)

        return {'status': "PASS" if result[0] else "FAIL", 'logs': logs, 'plots': result[2]}
 
    def _unique_value_check(self):

        result, logs = self._run_test(unique_value_check, self.train_df, self.oos_df, self.cat_cols_fixed)

        return {'status': "PASS" if result[0] else "FAIL", 'logs': logs, 'plots': result[2]}
 
    def _data_distribution_check(self):

        result, logs = self._run_test(data_distribution_check, self.train_df, self.oos_df, self.alt_model_features, self.not_considered_columns)

        return {'status': "PASS" if result[0] else "FAIL", 'logs': logs, 'dataframe_results': result[1]}
 