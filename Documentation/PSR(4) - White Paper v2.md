# PSR\(4\) \- White Paper v2

**Title: Payments Sales & Retention - ZAPP Edge(ACH+) Opt-In Prediction Model (Version 2.0)**

# Identifying Provider Propensity for ZAPP Edge(ACH+) Opt-In with Machine Learning Algorithm XGBClassifier

## Summary of Changes from Version 1.0

Since the initial release of the provider opt-in model, several key enhancements have been made to improve accuracy, interpretability, and business relevance.

- **Transition to a Binary Classification Modeling Approach**  
The underlying methodology was redesigned from a Random Survival Forest (a time-to-event survival model) to an XGBClassifier. This shift directly optimizes for the probability of ACH+ Opt-In within the observation window, producing well-calibrated propensity scores that are directly usable for provider-level ranking and business-tier segmentation.
- **Expanded Target Population**  
The pre-split dataset grew from **1,190,897** provider-observation records in Version 1.0 to **11,436,723** in Version 2.0 — a nearly **10× expansion**. This broader coverage exposes the model to a far wider range of provider behaviors and historical Opt-In signals, strengthening its ability to generalize across the eligible provider base..
- **Refined Feature Set**  
The feature set was refined based on Version 1.0 learnings. A correlation-grouping approach was applied — features were clustered by pairwise correlation strength, and from each cluster the feature with the strongest correlation to the target was retained. This produced a compact set of 12 features spanning payment behavior, opt-out history, interaction recency, and provider demographics, while eliminating multicollinearity that would otherwise dilute feature importance.
- **Updated Opt-in Driver Output Structure**  
The output structure has been redesigned to be directly actionable by the PSR team. Each scored provider now receives a **propensity probability**, a **five-tier category** (Very High → Very Low) derived from train-set cutoffs, per-category and global outreach rankings, and the **top 5 SHAP drivers** paired with the actual feature values in that row. This transforms the model's output from a raw risk score into an evidence-backed lead list, enabling sales conversations grounded in the specific behaviors and payment patterns that drove each provider's score.

## Abstract/Executive Summary

Version 2.0 of the Provider Opt-in Model introduces important enhancements to improve both predictive performance and business usability.

First, the definition of a “valid payment” was refined after the Business Unit (BU) and ZDS identified a payment status delay within the source database. Correcting this issue ensured that provider activity is now measured more accurately, strengthening the reliability of the model inputs. During this review process, the team also incorporated new data related to provider portal behavior. These features capture how providers interact with Zelis digitally and add meaningful behavioral signals that were not available in Version 1.0. From a modeling perspective, ZDS transitioned from survival analysis to a traditional machine learning approach (XGBClassifier). While survival analysis offers flexibility in modeling opt-in likelihood across multiple time horizons, the BU's primary use case focuses on identifying providers most likely to opt into ACH+. The updated model achieved strong performance, with an AUC of approximately 0.9 across training, validation, and out-of-sample datasets—indicating consistent and reliable discrimination between high and low propensity providers. Importantly, provider Opt-in remains a rare event with an overall opt-in percentage of 0.1%, the model addresses this by concentrating high-propensity providers into a small, prioritizable segment and pairing each score with the top drivers behind it — giving the PSR team a focused, evidence-backed outreach list rather than a broad, undifferentiated pool. 

## Introduction

The Zelis Payments Business Unit (BU) oversees two provider-paid product lines: ACH+ and Virtual Credit Card (VCC). Among these, ACH+ represents the highest-revenue product and a significant contributor to overall BU profitability. While the Opt-in rate to ACH+ is quite low (approximately 0.1%), the financial upside of even modest improvements in conversion is meaningful. Every additional Opt-in represents new recurring payment volume routed through ACH+, translating directly into incremental annualized revenue for the BU.

Prior to Version 1.0, the Payments Sales & Retention (PSR) team engaged providers who had not opted into ACH+ with limited prioritization, relying primarily on account age and annualized payment volume. Given the volume of eligible providers and the low natural Opt-In rate, this approach limited both sales efficiency and revenue capture. To address this gap, the BU partnered with Zelis Data Science (ZDS) to develop a predictive model that identifies providers most likely to opt into ACH+, enabling proactive and targeted outreach. Version 1.0 used a survival analysis framework (Random Survival Forest) to model time-to-Opt-In based on each provider's paper period. While it delivered acceptable baseline performance, Version 2.0 reevaluates the modeling approach to better align with the BU's operational need for a prioritized outreach list at each scoring cycle.

Version 2.0 introduces several enhancements. The modeling methodology has transitioned from survival analysis to a classification approach using XGBClassifier, selected for its strong predictive performance on tabular data and native support for per-row explainability via SHAP. Class imbalance is handled explicitly via RandomUnderSampler (1:200 negative-to-positive ratio) on the training set. The output structure has been redesigned to include a per-provider probability, a five-tier propensity category (Very High / High / Medium / Low / Very Low) derived from train-set cutoffs, per-category and global outreach rankings, and the top 5 SHAP drivers with their actual feature values — making each score directly actionable in a sales conversation.

Together, these enhancements improve predictive performance, interpretability, and alignment with the BU's primary objective: identifying which providers to prioritize for ACH+ outreach at each scoring cycle.

## Methodology

Version 2.0 reframes the problem from survival analysis (Random Survival Forest) to binary classification (XGBClassifier), producing a probability score between 0 and 1 for each provider. These scores are used to rank providers and group them into five propensity categories (Very High → Very Low), giving the PSR team a prioritized outreach list at each scoring cycle.

### **Data Collection**

Version 2.0 uses the same "snapshot" data structure introduced in Version 1.0, with refinements to the observation-period definition and eligibility filters to better align with the classification framework.

The dataset is built at the provider-observation-period level. Each row represents a single provider during a single "paper" period — the span from when the provider entered a Sales campaign (either from account creation or from a prior Opt-Out) up to their next action. Each row captures:

- The provider's demographic and account attributes at the start of the paper period 
-  Their payment activity, phone-interaction volume, and prior Opt-In/Opt-Out history during the paper period 
-  An indicator of whether the provider opted into ACH+ at the end of the paper period (target variable: OPT\_IN\_ACH)

Providers are uniquely identified within a row by the composite key **TIN + PROVIDERID + DATE\_START + DATE\_END.**

To align with the operational scoring window and support meaningful feature computation, the dataset is restricted to:

- Observations with Time\_Of\_Study \> 90 days
- Observations with at least one valid Check payment during the paper period
- Non-EPC serviced providers

### Feature Engineering

ZDS conducted an extensive exploration of over 70 features associated with a provider's tenure. These were developed to serve as potential predictors for the opt-in model. They are systematically categorized, as summarized below: 

| Category | Features |
| --- | --- |
| Payment Behavior | Sum check payment count (90d), check payment count momentum (30d vs 90d), average check payment (30d), cancelled check payment count, distinct check payer count |
| Opt-Out History | Total prior opt-outs (lifetime), total prior opt-outs (last year) |
| Interaction Recency | Time since last check payment, time since last cancelled check payment |
| Demographics | Provider type is Medical, first action indicator (new account vs Circle Back) |
| Phone | Phone interaction count (60d) |

Feature engineering steps included:

- **Missing value imputation** — 0 for count-based features and 9,999 for time-since features (representing "no event")
- **Derived momentum features** — e.g., ratio of 30-day check payment count to 90-day check payment count, capturing recent trend direction
- **Decision Tree-based binning** — used for high-cardinality categorical variables such as STATE and provider taxonomy
- **One-hot encoding** — applied to categorical variables (e.g., PROVIDERTYPE), fitted on training data and consistently applied to validation and OOS
- **Correlation analysis** — used to identify and manage multicollinearity among candidate features

### Model Development

As part of Version 2.0, ZDS evaluated two classification approaches on the refined dataset: Random Forest and XGBClassifier. Models were compared using top-3 decile capture percentage on the out-of-sample dataset as the primary criterion, with out-of-sample ROC-AUC as a tiebreaker. XGBClassifier outperformed Random Forest on both metrics and was selected as the final model, delivering the strongest concentration of Opt-Ins at the top of the ranked lead list along with clearer per-row explainability via SHAP.

#### Model Training & Feature Selection

To ensure robust model performance and proper evaluation, the dataset was divided into three distinct groups: Training, Validation, and Test (Out-of-Sample).

| Dataset | Time Period | Opt-In Percentage | Purpose |
| --- | --- | --- | --- |
| Training | 2022(Q4) – 2025(Q3) | 0.1% | Model development |
| Validation | 2025(Q4) | 0.15% | Model tuning and performance monitoring on unseen providers |
| Test (Out-of-Sample) | 2026-01-01 to 2026-02-01 | 0.08% | Future performance evaluation |

The Test dataset — also referred to as the out-of-sample dataset — represents a future time window. This dataset was held entirely separate from model development and was used to confirm that the model generalizes well to future Opt-In behavior. Across all datasets, the Opt-In rate remained low, reinforcing the challenge of predicting rare events and highlighting the importance of a high-performing model. To address this class imbalance during training, RandomUnderSampler was applied to the training set with a 1:200 negative-to-positive ratio — retaining all positive-class rows and under sampling the negatives to a workable size. Validation and Out-of-Sample datasets were left at their natural distribution to ensure evaluation metrics reflect real-world performance.

Feature selection was driven by a correlation-grouping approach. A correlation matrix was computed across all candidate features, and features with high pairwise correlation were clustered into groups representing shared underlying signals. From each group, the feature with the strongest correlation to the target variable was retained, and the remaining redundant features in the group were dropped. This eliminated multicollinearity while preserving distinct behavioral signals in the data. The final 12 features span payment activity, interaction recency, opt-out history, phone engagement, and provider type — capturing the key dimensions of Opt-In propensity while keeping the model parsimonious and its outputs interpretable for sales conversations.

| **Feature** | **Description** |
| --- | --- |
| SUMCHECKPAYMENTAMOUNT90​ | Payment amount in the past 90 days​ |
| CHECK\_PAYMENTCOUNT\_MOMENTUM\_30\_90​ | A payment count momentum variable that looks at the ratio between average payment count in the past 30 days and average payment count in the past 90 days​ |
| TIME\_SINCE\_LAST\_CHECK\_PAYMENT​ | Time (days) since last check payment​ |
| PROVIDERTYPE\_MEDICAL​ | Whether the provider type is medical or not​ |
| CHECK\_PAYMENT\_AVG\_30​ | The average payment (payment amount/count) in the past 30 days​ |
| PHONECOUNT60​ | The number of disposition phone counts in the last 60 days of the observation period​ |
| PAYERID\_CHECK\_COUNT​ | The number of payers associated with all check payments since last opt-out​ |
| TOTAL\_PREV\_OPTOUTS\_LAST\_YEAR​ | Total number of previous opt-outs in the last year​ |
| TOTAL\_PREV\_OPTOUTS​ | Total number of previous opt-outs ​ |
| TIME\_SINCE\_LAST\_CANCELLED\_CHECK\_PAYMENT​ | Time (days) since last cancelled check payment​ |
| FIRSTACTION\_BOOL​ | Whether the provider is an undecided (1) provider or a circle back provider (0). Can also be thought of as whether provider hasn’t had an opt-out​ |
| CANCELLEDCHECKPAYMENTCOUNT​ | The number of cancelled check payments​ |

## Results

The XGBClassifier champion model demonstrates strong, consistent discriminatory power across all three evaluation datasets, with particularly high concentration of opt-in propensity in the top score tiers.

### Performance Metrics

Model performance was evaluated across Training, Validation, and Out-of-Sample datasets using standard classification metrics with primary emphasis on ROC-AUC.

| Metric | Train | Validation | OOS |
| --- | --- | --- | --- |
| ROC-AUC | 0.8977 | 0.9231 | 0.8723 |
| Accuracy | 0.6812 | 0.678 | 0.6714 |
| Precision | 0.003 | 0.0046 | 0.0023 |
| Recall | 0.9512 | 0.9744 | 0.9268 |
| F1-Score | 0.006 | 0.0092 | 0.0046 |

**Decile-based** ranking analysis serves as the primary evaluation for the model, since the operational use case is generating a prioritized outreach list — success depends on how well the model concentrates true Opt-Ins in its top bins, not on threshold-based classification accuracy. Providers within each dataset (Training, Validation, and Out-of-Sample) were rank-ordered by their predicted ACH+ Opt-In probability and split into 10 equal-count bins.

For each decile bin, the following metrics were computed:

- **Positive Target: **number of providers who have opted-in.
- **Target Rate:** fraction of providers in the bin who opted into ACH+.
- **Overall Target PCT:** fraction of all Opt-Ins in the dataset captured by this decile.
- **Lift:** ratio of the decile's Opt-In rate to the overall dataset Opt-In rate.

The decile analysis demonstrates strong and consistent discriminatory power across all three datasets. Decile 1 captures the vast majority of actual Opt-Ins (**62%** in Training, **72%** in Validation, **46%** in OOS), and lift decreases monotonically from top to bottom deciles — confirming reliable rank-ordering by Opt-In propensity. The segment-wise view is equally strong: Decile 1 achieves a **3.75× lift** capturing **44%** of positives among Circle Backs and a **38.84× lift** capturing **57%** among Undecideds. These results confirm that the model is fit for its operational purpose: concentrating true Opt-Ins into a small, prioritized outreach segment.

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| ** Training Data Decile** | **Counts** | **Positive Target** | **Target Rate** | **Overall Target PCT** | **Lift** |
| 1 | 846329 | 5288 | 0.624816118 | 62.04388126 | 6.204386 |
| 2 | 846329 | 1854 | 0.219063745 | 21.75290391 | 2.17529 |
| 3 | 846329 | 874 | 0.103269532 | 10.25460519 | 1.02546 |
| 4 | 846329 | 313 | 0.036983254 | 3.672415816 | 0.367241 |
| 5 | 846329 | 112 | 0.013233624 | 1.314091282 | 0.131409 |
| 6 | 846329 | 48 | 0.005671553 | 0.563181978 | 0.056318 |
| 7 | 846329 | 28 | 0.003308406 | 0.328522821 | 0.032852 |
| 8 | 846328 | 6 | 0.000708945 | 0.070397747 | 0.00704 |
| 9 | 846328 | 0 | 0 | 0 | 0 |
| 10 | 846328 | 0 | 0 | 0 | 0 |

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| ** Validation Data Decile** | **Counts** | **Positive Target** | **Target Rate** | **Overall Target PCT** | **Lift** |
| 1 | 179456 | 2168 | 1.208095578 | 72.05051512 | 7.863646282 |
| 2 | 197800 | 526 | 0.265925177 | 17.48089066 | 1.730940471 |
| 3 | 208424 | 202 | 0.096917821 | 6.713193752 | 0.630850307 |
| 4 | 171207 | 77 | 0.044974797 | 2.558989698 | 0.292746616 |
| 5 | 125430 | 11 | 0.008769832 | 0.365569957 | 0.05708394 |
| 6 | 184495 | 7 | 0.003794141 | 0.232635427 | 0.02469654 |
| 7 | 221889 | 6 | 0.002704055 | 0.199401795 | 0.017601033 |
| 8 | 210469 | 2 | 0.000950259 | 0.066467265 | 0.006185354 |
| 9 | 210602 | 8 | 0.003798634 | 0.265869059 | 0.02472579 |
| 10 | 248824 | 2 | 0.000803781 | 0.066467265 | 0.005231912 |

|  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- |
| ** Test (OOS) Data Decile** | **Counts** | **Positive Target** | **Target Rate** | **Overall Target PCT** | **Lift** |
| 1 | 81245 | 382 | 0.47018278 | 45.85834334 | 5.728214801 |
| 2 | 106284 | 235 | 0.221105717 | 28.21128451 | 2.693720595 |
| 3 | 119630 | 142 | 0.118699323 | 17.04681873 | 1.446108294 |
| 4 | 99971 | 56 | 0.056016245 | 6.722689076 | 0.682443287 |
| 5 | 59138 | 5 | 0.008454801 | 0.600240096 | 0.10300444 |
| 6 | 84413 | 1 | 0.001184652 | 0.120048019 | 0.014432556 |
| 7 | 112632 | 6 | 0.005327083 | 0.720288115 | 0.064899601 |
| 8 | 109561 | 2 | 0.001825467 | 0.240096038 | 0.02223958 |
| 9 | 110127 | 4 | 0.00363217 | 0.480192077 | 0.044250559 |
| 10 | 131839 | 0 | 0 | 0 | 0 |

The analysis demonstrates strong discriminatory power, with Opt-In rates rising consistently as predicted propensity increases. Higher-propensity segments experience meaningfully higher Opt-In rates than lower-propensity segments, confirming that the model effectively differentiates providers by their likelihood of opting into ACH+

## Segment-Wise Decile Analysis(Circle Backs and Undecideds)

The two segments can be defined based on their opt-in history, in the following way:

- **Undecideds** — Providers active in a Sales campaign for over 90 days without taking any action (no Opt-In, no Opt-Out). The observation begins at account creation.
- **Circle Backs** — Providers who previously Opted-Out of a paid program and are eligible for re-engagement after a waiting period. The observation begins after a prior Opt-Out.

The model is evaluated separately by provider segment (Circle Backs: FIRSTACTION\_BOOL=0; Undecideds: FIRSTACTION\_BOOL=1) to understand how scoring performance distributes across the two target populations.

#### OOS Decile 1 Reference Table

| Metric | Circle Backs | Undecideds |
| --- | --- | --- |
| Segment size | 640451 | 374389 |
| Segment avg opt-in rate | 0.115% | 0.024% |
| Decile 1 count | 75752 | 5493 |
| Decile 1 opt-in rate | 0.434% | 0.964% |
| Decile 1 lift (segment baseline) | 3.75x | 38.84x |
| % of segment positives in Decile 1 | 44.45% | 56.98% |

### **Performance Comparison with Production Model**

The retrained XGBClassifier model delivers substantially stronger lift than the Random Survival Forest production model across all provider segments. This confirms that Version 2.0 improves overall discrimination.

|  |  |  |
| --- | --- | --- |
| Metric​ | Production Model​ | Retrained Model​ |
| Algorithm​ | Random Survival Forest​ | XGBClassifier​ |
| Overall Lift (OOS)​ | 1.58x​ | 5.72x​ |
| Circle Back Lift (OOS)​ | 1.92x​ | 3.75x​ |
| Undecided Lift (OOS)​ | 1.54x​ | 38.84x​ |

## **Opt-In Categorization**

To make model outputs actionable for the PSR team, predicted probabilities were converted into five opt-in categories based on rank ordering from the training dataset. These thresholds were then consistently applied across the Training, Validation, and Out-of-Sample (OOS) datasets.

|  |  |  |  |  |
| --- | --- | --- | --- | --- |
| Opt-In Category​ | Approx. Proportion of Data​ | Approx. Opt-In Rate Within Category​ | Total Pct of Opt-Ins Captured​​ | Lift​ |
| Very High​ | \~2%​ | 0.82%​ | 11.52%​ | 10.36x​ |
| High​ | \~8%​ | 0.41%​ | 34.33%​ | 5.13x​ |
| Medium​ | \~17%​ | 0.17%​ | 40.94%​ | 2.23x​ |
| Low​ | \~17%​ | 0.06%​ | 11.40%​ | 0.77x​ |
| Very Low​ | \~56%​ | 0.002%​ | 1.80%​ | 0.03x​ |

The propensity categorization delivers strong stratification of Opt-In likelihood across the scored population. Lift decreases monotonically from the **Very High** tier down to **Very Low**, confirming that the categories are reliably ordered by Opt-In propensity. The top two tiers (**Very High** and **High**) together represent a small share of the population but capture a disproportionate share of actual Opt-Ins, making them the operational sweet spot for focused sales outreach. The **Medium** tier maintains above-baseline lift and offers a secondary layer for expanded campaigns when resource capacity allows. In contrast, the **Low** and **Very Low** tiers cover the bulk of the population but contribute very few Opt-Ins — confirming that the model effectively de-prioritizes providers unlikely to convert. This clean gradient validates the category cutoffs as an actionable prioritization framework for the PSR team.

### Business Impact Analysis

To evaluate how the BU can operationalize the model, a tiered outreach analysis was conducted to simulate ACH+ opt-in campaign scenarios using Q1 2026 OOS results.

**Scenario 1: Very High Tier Only (Top \~2% of Providers)**

Focus outreach exclusively on the highest-propensity providers.

- **Target List Size:** \~20,297 providers (\~2% of OOS population)
- **Capture Rate:** 11.52% of all expected Opt-Ins

**Scenario 2: Very High + High Tiers (Top \~10% of Providers)**

Expand outreach to cover both the Very High and High propensity tiers.

- **Target List Size:** \~101,484 providers (\~10% of OOS population)
- **Capture Rate:** 45.85% of all expected Opt-Ins (11.52% + 34.33%)

**Scenario 3: Very High + High + Medium Tiers (Top \~27% of Providers)**

Capture the vast majority of Opt-In opportunity.

- **Target List Size:** \~274,007 providers (\~27% of OOS population)
- **Capture Rate:** 86.79% of all expected Opt-Ins (11.52% + 34.33% + 40.94%)

## Comparison: Version 1.0 vs. Version 2.0

| Dimension | Version 1.0 | Version 2.0 | Change |
| --- | --- | --- | --- |
| Algorithm | Random Survival Forest | XGBClassifier | Methodology redesign |
| Scored Population | 356833 | \~1M+ (full eligible population) | 10x+ broader coverage |
| Base Opt-In Rate | 2.22% | 0.1% (full population) | Reflects true population base rate |
| Primary Metric | C-index (survival) | ROC-AUC (classification) | Directly optimized for ranking |
| Lift (top decile, OOS) | 1.58x overall (1.92x Circle Backs; 1.54x Undecideds) | 5.72x overall (3.75x Circle Backs; 38.84x Undecideds) | \~4.14x overall improvement; Undecideds lift improved dramatically |
| Feature Count | 12 | 12 | Different sets of features |
| Training Records | \~833K (observation periods) | 8,463,287 (provider snapshots) | 10x+ larger training data |

---

## Conclusion

Version 2.0 of the PSR4 ACH+ Opt-In Propensity Model delivers a fundamental improvement in how the PSR team can prioritize outreach to Circle Back and Undecided providers. By transitioning from a survival-analysis framework (Random Survival Forest) to an XGBClassifier, the model now enables a genuinely data-driven, rank-ordered prioritization strategy directly aligned with the BU's operational need for a prioritized outreach list at each scoring cycle.

The champion model — XGBClassifier trained on a 1:200 under sampled dataset — outperforms Random Forest on the primary selection criterion (top-3 decile capture percentage) and exceeds it on out-of-sample ROC-AUC. Compared to Version 1.0, the retrained model delivers materially stronger lift across all provider segments. The top propensity tier concentrates a disproportionate share of actual Opt-Ins into a small, highly actionable outreach segment.

The model's top predictive signals — including time since last check payment, check payment momentum (30-day vs 90-day), average check payment amount, distinct payer count, and prior opt-out history — are consistent with business intuition and stable across all evaluation datasets. Per-row explainability via SHAP now surfaces the top 5 drivers behind each provider's score alongside the actual feature values, giving the sales team a concrete, evidence-backed narrative to enter every conversation with — a capability that was not available in Version 1.0.

Version 2.0 provides the BU with a scalable, interpretable, and high-ROI tool for proactive ACH+ Opt-In campaigns — concentrating outreach resources where expected conversions are highest, and enabling the PSR team to pursue ACH+ enrollment with confidence in the underlying model signal.

## Appendix

| **Description of File** | **Location** | **Additional Comments** |
| --- | --- | --- |
|  |  |  |
|  |  |  |
