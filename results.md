# Model comparison results

These results cover the September 10 and September 22 milestones and were generated during the October 3, 2026 completion session. Positive means income above $50K. Metrics use a fixed probability threshold of 0.50 and give each record equal weight.

## Final test comparison

| model | n | accuracy | precision | recall | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Majority class | 16281 | 0.7638 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2362 |
| Logistic regression | 16281 | 0.8487 | 0.7204 | 0.5876 | 0.6473 | 0.9025 | 0.7552 |
| Random forest | 16281 | 0.8649 | 0.7839 | 0.5913 | 0.6741 | 0.9157 | 0.7979 |

The random forest improves F1 by 0.0268 over logistic regression. Its precision is 78.4%, while recall is 59.1%. Accuracy alone hides missed higher-income records. F1 balances precision and recall for the minority positive class; it excludes true negatives, so accuracy, both error rates and probability-ranking metrics are reported alongside it. There is no specified real-world error-cost ratio, so F1 is a transparent classroom choice, not a claim of optimal business utility.

## Validation and model selection

The development file is divided into 26,049 training and 6,512 validation records using a seeded five-fold stratified group split, taking the first fold as validation. Identical raw predictor records stay in one group. Both models use the same records and features. The supplied 16,281-row test file is evaluated only after choosing forest depth on validation F1. The forest uses 200 trees, minimum leaf size 3 and selected maximum depth unlimited. The candidate depths are 12, 20 and unlimited. The threshold is fixed at 0.5, not tuned. Models are not refitted on validation data.

Validation search (`outputs/validation_search.csv`, generated when the analysis runs) records every candidate. Imputation, numeric scaling and category vocabularies are fitted only on training records. All missing-data rows remain. No test performance is used to select the model or threshold; test failure inspection is post-evaluation analysis.

## Training and validation checks

| model | split | n | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- |
| Majority class | train | 26049 | 0.7592 | 0.0000 | 0.0000 | 0.0000 |
| Majority class | validation | 6512 | 0.7592 | 0.0000 | 0.0000 | 0.0000 |
| Logistic regression | train | 26049 | 0.8489 | 0.7297 | 0.5916 | 0.6534 |
| Logistic regression | validation | 6512 | 0.8606 | 0.7531 | 0.6263 | 0.6838 |
| Random forest | train | 26049 | 0.8791 | 0.8281 | 0.6282 | 0.7145 |
| Random forest | validation | 6512 | 0.8745 | 0.8106 | 0.6250 | 0.7058 |

The train-to-validation gap indicates how much performance drops outside the fitting data. These are point estimates from one split, not proof of a statistically significant improvement. The original notebooks had already examined parts of the development data, so this is a repaired evaluation protocol rather than a preregistered experiment.

## Confusion matrices

Rows are actual labels and columns are predicted labels. In each row below, TN means actual <=50K predicted <=50K; FP means actual <=50K predicted >50K; FN means actual >50K predicted <=50K; TP means actual >50K predicted >50K.

| model | tn | fp | fn | tp |
| --- | --- | --- | --- | --- |
| Majority class | 12435 | 0 | 3846 | 0 |
| Logistic regression | 11558 | 877 | 1586 | 2260 |
| Random forest | 11808 | 627 | 1572 | 2274 |

All matrices total 16,281. Raw predictions (`outputs/test_predictions.csv`, generated when the analysis runs) permit independent recalculation. Run manifest (`outputs/run_manifest.json`, generated when the analysis runs) records versions, seeds, data hashes and checks.

##Individual Prediction
| Feature | Value | Contribution |
|---|---|---|
| age | 28 | -0.10842 |
| marital_status | Married-civ-spouse | +0.09789 |
| relationship | Husband | +0.07300 |
| workclass | Local-gov | +0.06953 |
| occupation | Protective-serv | +0.05181 |
| capital_gain | 0 | -0.02459 |
| hours_per_week | 40 | -0.01703 |
| sex | Male | +0.01429 |
| education_num | 12 | +0.01170 |
| capital_loss | 0 | -0.00899 |
| native_country | United-States | +0.00207 |
| race | White | +0.00139 |
| **Total** | | **+0.16265** |
