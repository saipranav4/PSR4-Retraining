import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import stats

import io 

import seaborn as sns
 
# --- Core Check Functions (2-Dataset Architecture: Train vs OOS) ---
 
def _compare_schemas(df1, df2, comp_name):

    """Internal helper for schema check to compare two dataframes."""

    df1_cols = set(df1.columns)

    df2_cols = set(df2.columns)
 
    extra_cols = df2_cols - df1_cols

    missed_cols = df1_cols - df2_cols
 
    failures = []

    failed_cols_set = set()
 
    if extra_cols:

        failures.append(f"Extra columns in {comp_name.split(' vs ')[1]}: {list(extra_cols)}")

        failed_cols_set.update(extra_cols)

    if missed_cols:

        failures.append(f"Missing columns in {comp_name.split(' vs ')[1]}: {list(missed_cols)}")

        failed_cols_set.update(missed_cols)
 
    return failures, failed_cols_set
 
def schema_check(train_df, oos_df):

    """Checks for missing or extra columns between Train and OOS."""

    failed_comparisons = {}

    all_failed_cols = set()
 
    failures, cols = _compare_schemas(train_df, oos_df, 'Train vs OOS')

    if failures:

        failed_comparisons['Train vs OOS'] = failures

        all_failed_cols.update(cols)
 
    all_pass = not failed_comparisons

    passed_comparisons = ['Train vs OOS'] if all_pass else []
 
    return all_pass, failed_comparisons, passed_comparisons, list(all_failed_cols)
 
def data_type_check(train_df, oos_df):

    """Checks if the data types of common columns are strictly consistent."""

    failed_dict = {}

    failures = []

    common_cols = set(train_df.columns) & set(oos_df.columns)
 
    for col in common_cols:

        dtype1 = train_df[col].dtype

        dtype2 = oos_df[col].dtype

        if dtype1 != dtype2:

            failures.append(f"Col '{col}': Train={dtype1}, OOS={dtype2}")
 
    if failures:

        failed_dict['Train vs OOS'] = failures

        print(f"⚠️ Mismatches found in Train vs OOS:\n" + "\n".join(failures))
 
    all_pass = not bool(failed_dict)

    return all_pass, failed_dict
 
def duplicate_check(train_df, oos_df, subset_cols=None):

    """Checks for duplicates in each dataframe."""

    dfs = {'Train': train_df, 'OOS': oos_df}

    all_pass = True

    logs = "Duplicate Check Report:\n"

    if subset_cols:

        logs += f"Checking duplicates based on intended subset: {subset_cols}\n"

    else:

        logs += "Checking duplicates based on all columns.\n"
 
    for name, df in dfs.items():

        dup_count = 0

        log_subset = ""

        try:

            if subset_cols:

                valid_subset = [col for col in subset_cols if col in df.columns]

                if len(valid_subset) != len(subset_cols):

                    missing = set(subset_cols) - set(df.columns)

                    logs += f"  - {name}: FAILED. Missing subset columns: {list(missing)}\n"

                    all_pass = False

                    continue 

                dup_count = df.duplicated(subset=valid_subset).sum()

                log_subset = f" (subset: {valid_subset})"

            else:

                dup_count = df.duplicated().sum()

                log_subset = " (all columns)"

        except Exception as e:

            logs += f"  - {name}: FAILED. Error: {e}\n"

            all_pass = False

            continue
 
        percent = (dup_count / len(df) * 100) if len(df) > 0 else 0

        if dup_count > 0:

            all_pass = False

            logs += f"  - {name}: FAILED. Found {dup_count} duplicate rows{log_subset} ({percent:.4f}%).\n"

        else:

            logs += f"  - {name}: PASSED.{log_subset}\n"
 
    return all_pass, logs
 
def missing_value_check(train_df, oos_df, not_considered_columns):

    """Perform missing value consistency checks between Train and OOS."""

    failed_cols = {}

    plots = []

    ref_cols = set(train_df.columns) - set(not_considered_columns)

    cmp_cols = set(oos_df.columns) - set(not_considered_columns)

    common_cols = list(ref_cols & cmp_cols)
 
    if not common_cols:

        return True, {}, []
 
    ref_missing_pct = train_df[common_cols].isna().mean() * 100

    cmp_missing_pct = oos_df[common_cols].isna().mean() * 100
 
    # Rule 1: <0.05%

    zero_null_cols = ref_missing_pct[ref_missing_pct < 0.05].index.tolist()

    for col in zero_null_cols:

        cmp_pct = cmp_missing_pct.get(col, 0)

        if cmp_pct > 0.05:

            failed_cols[col] = f"[{col}] <0.05% missing in Train but {cmp_pct:.2f}% in OOS"
 
    # Rule 2: 0.05-70% (+/- 10%)

    mid_null_cols = ref_missing_pct[(ref_missing_pct >= 0.05) & (ref_missing_pct < 70)].index.tolist()

    for col in mid_null_cols:

        ref_pct = ref_missing_pct[col]

        cmp_pct = cmp_missing_pct.get(col, 0)

        lower, upper = ref_pct - 10, ref_pct + 10

        if not (lower <= cmp_pct <= upper):

            failed_cols[col] = f"[{col}] Train null% = {ref_pct:.2f}%, OOS null% = {cmp_pct:.2f}% (Limit: [{lower:.2f}%, {upper:.2f}%])"
 
    all_pass = not bool(failed_cols)

    failed_summary = {'Train vs OOS': list(failed_cols.values())} if failed_cols else {}
 
    if failed_cols:

        comparison_df = pd.DataFrame({

            'Train': ref_missing_pct.loc[list(failed_cols.keys())], 

            'OOS': cmp_missing_pct.loc[list(failed_cols.keys())]

        })

        fig, ax = plt.subplots(figsize=(8, 5)) 

        comparison_df.plot(kind='bar', ax=ax)

        for container in ax.containers:

            ax.bar_label(container, fmt='%.2f%%', label_type='edge', fontsize=8, padding=3)

        ax.set_title('Missing Value Percentage: Train vs OOS')

        ax.set_ylabel('Percentage of Missing Rows')

        plt.xticks(rotation=45, ha='right', fontsize=7)

        plt.tight_layout()
 
        buf = io.BytesIO()

        plt.savefig(buf, format='png')

        plt.close(fig)

        buf.seek(0)

        plots.append(buf)
 
    return all_pass, failed_summary, plots
 
def long_tail_test(train_df, oos_df, columns):

    all_pass = True

    failure_summary = {}

    plots = []

    head_threshold = 0.03 

    tolerance = 0.1
 
    print("\n--- Running Long Tail Distribution Checks ---")

    for col in columns:

        if col not in train_df.columns or col not in oos_df.columns:

            continue
 
        dist1 = train_df[col].value_counts(normalize=True)

        dist2 = oos_df[col].value_counts(normalize=True)

        head_categories_1 = dist1[dist1 > head_threshold]

        failed = False

        for cat, pct1 in head_categories_1.items():

            pct2 = dist2.get(cat, 0)

            if abs(pct1 - pct2) > tolerance:

                failed = True
 
        tail_pct1 = dist1[dist1 <= head_threshold].sum()

        tail_pct2 = dist2.loc[dist2.index.intersection(dist1[dist1 <= head_threshold].index)].sum()

        if abs(tail_pct1 - tail_pct2) > tolerance:

            failed = True
 
        if failed:

            all_pass = False

            failure_summary.setdefault('Train vs OOS', []).append(col)

            # Plot

            all_head_cats = sorted(list(set(head_categories_1.index) | set(dist2[dist2 > head_threshold].index)))

            tail_names = dist1.index.difference(all_head_cats).union(dist2.index.difference(all_head_cats))

            t1 = dist1.loc[dist1.index.intersection(tail_names)].sum()

            t2 = dist2.loc[dist2.index.intersection(tail_names)].sum()

            plot_data = {

                'Train': [dist1.get(c, 0) for c in all_head_cats] + [t1],

                'OOS': [dist2.get(c, 0) for c in all_head_cats] + [t2]

            }

            df_plot = pd.DataFrame(plot_data, index=all_head_cats + ['Tail']).T.reset_index().melt('index', var_name='Category', value_name='Percentage')

            fig, ax = plt.subplots(figsize=(10, 6))

            sns.barplot(data=df_plot, x='Category', y='Percentage', hue='index', ax=ax, palette='viridis')

            ax.set_title(f"Distribution Comparison for '{col}'")

            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()

            buf = io.BytesIO()

            plt.savefig(buf, format='png')

            plt.close(fig)

            buf.seek(0)

            plots.append(buf)
 
    return all_pass, failure_summary, plots
 
def length_distribution_with_long_tail(train_df, oos_df, columns):

    head_threshold = 0.05 

    tolerance = 0.10

    all_pass = True

    failed_cols = {}

    plots = []
 
    for col in columns:

        if col not in train_df.columns or col not in oos_df.columns: continue

        dist1 = train_df[col].astype(str).str.len().value_counts(normalize=True).sort_index()

        dist2 = oos_df[col].astype(str).str.len().value_counts(normalize=True).sort_index()

        head_lengths = dist1[dist1 > head_threshold]

        tail_lengths = dist1[dist1 <= head_threshold]

        passed = True

        for length, prop1 in head_lengths.items():

            if abs(prop1 - dist2.get(length, 0)) > tolerance: passed = False

        if abs(tail_lengths.sum() - dist2.loc[dist2.index.intersection(tail_lengths.index)].sum()) > tolerance:

            passed = False
 
        if not passed:

            all_pass = False

            failed_cols.setdefault('Train vs OOS', []).append(col)

            all_len = sorted(set(dist1.index).union(dist2.index))

            x = np.arange(len(all_len))

            fig, ax = plt.subplots(figsize=(8, 5))

            ax.bar(x - 0.175, [dist1.get(l, 0) for l in all_len], 0.35, label='Train')

            ax.bar(x + 0.175, [dist2.get(l, 0) for l in all_len], 0.35, label='OOS')

            ax.set_title(f'Length Distribution for {col}')

            ax.set_xticks(x)

            ax.set_xticklabels(all_len)

            ax.legend()

            plt.tight_layout()

            buf = io.BytesIO()

            plt.savefig(buf, format='png')

            plt.close(fig)

            buf.seek(0)

            plots.append(buf)
 
    return all_pass, failed_cols, plots
 
def outlier_check(train_df, oos_df, columns):

    tolerance = 10

    all_pass = True

    detailed_failed_checks = {}

    all_plots = []
 
    common_cols = list(set(train_df.columns) & set(oos_df.columns) & set(columns))

    failed_columns = []

    for col in common_cols:

        ref_s = train_df[col]

        cmp_s = oos_df[col]

        ref_clean = ref_s.dropna()

        q1, q3 = ref_clean.quantile(0.10), ref_clean.quantile(0.90)

        iqr = q3 - q1

        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
 
        def get_pcts(s):

            tot = len(s)

            if tot == 0: return 0, 0, 0, 0

            sn = s.dropna()

            return (s.isna().sum()/tot*100, (sn<lower).sum()/tot*100, 

                    ((sn>=lower)&(sn<=upper)).sum()/tot*100, (sn>upper).sum()/tot*100)
 
        r_nul, r_bel, r_wit, r_abo = get_pcts(ref_s)

        c_nul, c_bel, c_wit, c_abo = get_pcts(cmp_s)
 
        failed_cats = []

        for name, r_val, c_val in [('nulls', r_nul, c_nul), ('below', r_bel, c_bel), ('within', r_wit, c_wit), ('above', r_abo, c_abo)]:

            if abs(r_val - c_val) > tolerance:

                failed_cats.append(name)
 
        if failed_cats:

            all_pass = False

            failed_columns.append(col)

            detailed_failed_checks.setdefault('Train vs OOS', {})[col] = failed_cats

            fig, ax = plt.subplots(figsize=(8, 4))

            x = np.arange(4)

            ax.bar(x - 0.175, [r_nul, r_bel, r_wit, r_abo], 0.35, label='Train')

            ax.bar(x + 0.175, [c_nul, c_bel, c_wit, c_abo], 0.35, label='OOS')

            ax.set_xticks(x, ['Nulls', '< Lower', 'Within', '> Upper'])

            ax.set_title(f'Outlier Dist for {col}')

            ax.legend()

            plt.tight_layout()

            buf = io.BytesIO()

            plt.savefig(buf, format='png')

            plt.close(fig)

            buf.seek(0)

            all_plots.append(buf)
 
    return all_pass, detailed_failed_checks, all_plots
 
def unique_value_check(train_df, oos_df, categorical_cols_to_check):

    tolerance = 0.1

    all_pass = True

    internal_failures = {}

    plots = []
 
    for col in categorical_cols_to_check:

        if col not in train_df.columns or col not in oos_df.columns: continue

        s1 = train_df[col].astype(str).str.strip().str.upper()

        s2 = oos_df[col].astype(str).str.strip().str.upper()

        c1 = s1.value_counts(normalize=True)

        c2 = s2.value_counts(normalize=True)

        col_fails = []

        for key in set(c1.index) | set(c2.index):

            if abs(c1.get(key, 0) - c2.get(key, 0)) > tolerance:

                col_fails.append(f"Shift in '{key}'")

        if col_fails:

            all_pass = False

            internal_failures.setdefault('Train vs OOS', []).append(col)

            plot_df = pd.DataFrame([{'Value': k, 'Prop': c1.get(k,0), 'Data': 'Train'} for k in c1.index[:10]] + 

                                   [{'Value': k, 'Prop': c2.get(k,0), 'Data': 'OOS'} for k in c2.index[:10]])

            fig, ax = plt.subplots(figsize=(8, 4))

            sns.barplot(data=plot_df, x='Value', y='Prop', hue='Data', ax=ax)

            ax.set_title(f"Unique Value Dist: {col}")

            plt.xticks(rotation=45)

            plt.tight_layout()

            buf = io.BytesIO()

            plt.savefig(buf, format='png')

            plt.close(fig)

            buf.seek(0)

            plots.append(buf)
 
    return all_pass, internal_failures, plots
 
def data_distribution_check(train_df, oos_df, alt_model2_features, not_considered_columns=None):

    tolerance = 0.1

    if not_considered_columns is None: not_considered_columns = []

    all_pass = True

    dfs = {}

    detailed_failed_checks = {}
 
    common_cols = [c for c in train_df.columns if c in oos_df.columns and 

                   pd.api.types.is_numeric_dtype(train_df[c]) and 

                   c not in not_considered_columns]
 
    rows, comp_fails = [], []

    for col in common_cols:

        v1_mean, v1_med = train_df[col].mean(), train_df[col].median()

        v2_mean, v2_med = oos_df[col].mean(), oos_df[col].median()
 
        for stat, v1, v2 in [('Mean', v1_mean, v2_mean), ('Median', v1_med, v2_med)]:

            if pd.isna(v1) or pd.isna(v2): continue

            if abs(v1) > 0 and (abs(v2 - v1) / abs(v1)) > tolerance:

                all_pass = False

                comp_fails.append({'col': col, 'metric': stat, 'ref_val': round(v1,2), 'cmp_val': round(v2,2)})

                rows.append({'Column': col, 'Feature': 'Yes' if col in alt_model2_features else 'No',

                             'Statistic': stat, 'Train': round(v1,2), 'OOS': round(v2,2), 

                             'Percent Diff': ((v2-v1)/v1)*100})
 
    if rows:

        df_result = pd.DataFrame(rows).sort_values(by=['Feature', 'Percent Diff'], ascending=[False, False])

        dfs['Train vs OOS'] = df_result

    if comp_fails:

        detailed_failed_checks['Train vs OOS'] = comp_fails
 
    return all_pass, dfs, detailed_failed_checks
 