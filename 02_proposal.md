# Adult income prediction project proposal

## Problem and dataset

Predict whether an individual's annual income is above $50K (positive class, 1) or at/below $50K (0). One record represents one person. This is binary classification of historical income, not a causal model of pay or an assessment of a person's worth. The local UCI Adult files contain 32,561 development and 16,281 test records, totaling 48,842.

## Feature choices and rationale

These are modeling judgments informed by the [UCI dataset description](https://archive.ics.uci.edu/dataset/2/adult) and the bundled [adult.names](DSDA310/adult.names), rather than claims that these choices must maximize accuracy.

| Feature | Decision and rationale |
| --- | --- |
| age | Keep; career stage can be associated with income. |
| education_num | Keep as an ordered education-level code; it is not literal years in school. |
| education | Exclude from the model to avoid including a second encoding of the same education level. Retain in the raw data for readable case descriptions. |
| workclass and occupation | Keep; employment sector and job category may carry income information. |
| hours_per_week | Keep; labor participation can be associated with earnings. |
| marital_status and relationship | Keep for the benchmark, with explicit recognition that these may proxy demographic information. |
| capital_gain and capital_loss | Keep in the full benchmark and remove together in an availability-risk ablation. Their admissibility depends on prediction timing. |
| sex and race | Include in the full academic benchmark so dependence can be audited; compare with a retrained model excluding both. Retain raw values for group evaluation either way. |
| native_country | Keep as a categorical context variable, while recognizing possible demographic proxy effects and small categories. |
| fnlwgt | Exclude as a predictor. It is a survey weight, not a person's income or an intrinsic personal characteristic. Report unweighted record-level performance, not population-weighted estimates. |
| income | Target only; never an input. |

## Cleaning and validation design

Convert `?` to missing values, strip whitespace, skip the test header and remove the test labels' trailing periods. Keep all records. Fit numeric median imputation, categorical mode imputation, numeric standardization and one-hot vocabularies inside training pipelines. Unseen categories are ignored by the encoder.

Create a roughly 80/20 development split using the first fold of a seeded five-fold stratified group splitter. Group identical raw predictor tuples so duplicates do not cross training and validation. Record split membership. Compare both model families on exactly the same records. Reserve the provided `adult.test` file for final evaluation after model selection. Audit raw predictor overlap between development and test and report a nonoverlap sensitivity result as well. No person ID is available, so identical rows are a proxy for possible duplication.

## Models and metric

Fit a majority-class reference and logistic regression baseline. Use a random forest as the stronger candidate because it can represent nonlinear relationships and interactions without manually specifying each one. Compare maximum depths 12, 20 and unlimited using validation F1, holding 200 trees and a minimum leaf size of 3 fixed. Use probability >=0.5 for all classifications and explanations. Do not tune the threshold or refit after seeing test results.

Use positive-class F1 as the primary metric because higher-income records are the minority and both false positives and false negatives matter in this educational comparison. Report accuracy, precision, recall, ROC AUC, average precision and confusion matrices alongside F1. Without a defined application and error costs, no metric establishes deployment suitability.

## Leakage and fairness risks

The specific availability-risk hypothesis added during this revision is that same-year capital gains/losses may not be known when predicting annual income in advance. Remove both fields, retrain on identical records, and measure the change. This is a conditional leakage risk, not a claim that UCI's fields are intrinsically invalid.

Audit sex and race groups using sample sizes, positive counts, selection rate, recall and false-positive rate. Add recall uncertainty intervals and compare the full model against a sex/race removal model. Removing protected attributes does not remove correlated proxies. Explain at least one wrong prediction and keep conclusions descriptive.
