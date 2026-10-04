# Evaluation and failure analysis

This report completes the September 29 analysis scope using measured results. It was produced on October 3, 2026; it does not claim the work was completed by the milestone date.

## Leakage risk and actual removal test

The original proposal stated that no major leakage was seen; it did not name a specific risky feature. The risk below is a newly documented completion-session hypothesis, not a retroactively claimed September 3 decision.

Assumption: if income is predicted before year-end, same-year capital gains and losses may be unavailable or too closely tied to the outcome period. They are legitimate benchmark inputs when available at prediction time; their predictive value alone does not prove leakage. We retrained the same forest configuration on the same training records after removing both fields, retaining the 0.5 threshold. No replacement feature or hyperparameter was selected using test results.

| model | split | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- |
| Full features | validation | 0.8745 | 0.8106 | 0.6250 | 0.7058 |
| Full features | test | 0.8649 | 0.7839 | 0.5913 | 0.6741 |
| Without capital gains and losses | validation | 0.8521 | 0.7494 | 0.5797 | 0.6537 |
| Without capital gains and losses | test | 0.8385 | 0.7103 | 0.5343 | 0.6099 |
| Without sex and race | validation | 0.8750 | 0.8060 | 0.6333 | 0.7093 |
| Without sex and race | test | 0.8647 | 0.7814 | 0.5931 | 0.6744 |

Removing capital fields changes test F1 by -0.0642. This measures dependence on the fields, not proof of their timing or causal relationship to income. If prediction-time availability cannot be established, use the reduced-input specification and evaluate it for the actual use case. The target is excluded from all feature lists. All fitted preprocessing uses training records only.

## Duplicate and overlap robustness

The development data contains 25 repeated raw predictor rows. Grouping prevents these identical predictor records from crossing training and validation. There are 26 official test records whose full raw predictor tuple also appears in development. We retain the supplied benchmark for the primary result and additionally score the nonoverlapping records: n=16,255, accuracy=0.8650, F1=0.6743. This does not establish that remaining records are independent people; there is no person identifier. See overlap sensitivity (`outputs/overlap_sensitivity.csv`, generated when the analysis runs).

## Group performance

All rates below are measured on the same official test file. Base rate is the actual higher-income share; selection rate is the predicted higher-income share; recall is TP/(TP+FN); false-positive rate is FP/(FP+TN). Recall intervals are approximate 95% Wilson intervals for each group, not confidence intervals for between-group differences. NA means a rate has no denominator.

| attribute | group | n | positives | base_rate | selection_rate | precision | recall | fpr | recall_ci_low | recall_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sex | Female | 5421 | 590 | 0.1088 | 0.0701 | 0.7921 | 0.5102 | 0.0164 | 0.4699 | 0.5503 |
| sex | Male | 10860 | 3256 | 0.2998 | 0.2321 | 0.7826 | 0.6060 | 0.0721 | 0.5891 | 0.6226 |
| race | Amer-Indian-Eskimo | 159 | 19 | 0.1195 | 0.0503 | 0.7500 | 0.3158 | 0.0143 | 0.1536 | 0.5399 |
| race | Asian-Pac-Islander | 480 | 133 | 0.2771 | 0.2208 | 0.7642 | 0.6090 | 0.0720 | 0.5241 | 0.6878 |
| race | Black | 1561 | 179 | 0.1147 | 0.0737 | 0.7826 | 0.5028 | 0.0181 | 0.4303 | 0.5752 |
| race | Other | 135 | 25 | 0.1852 | 0.0815 | 0.9091 | 0.4000 | 0.0091 | 0.2340 | 0.5926 |
| race | White | 13946 | 3490 | 0.2503 | 0.1908 | 0.7843 | 0.5980 | 0.0549 | 0.5816 | 0.6141 |

Male recall is 60.6% and female recall is 51.0%, a signed male-minus-female gap of 9.58 percentage points. This shows unequal detection of higher-income individuals in this sample. It does not establish discrimination as a causal conclusion. Race-group denominators and intervals matter: small positive counts make recall estimates unstable. No blanket fairness pass is asserted.

## Removing sex and race

| attribute | group | n | positives | selection_rate | recall | fpr | recall_ci_low | recall_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sex | Female | 5421 | 590 | 0.0697 | 0.5102 | 0.0159 | 0.4699 | 0.5503 |
| sex | Male | 10860 | 3256 | 0.2340 | 0.6081 | 0.0738 | 0.5912 | 0.6247 |
| race | Amer-Indian-Eskimo | 159 | 19 | 0.0692 | 0.4211 | 0.0214 | 0.2314 | 0.6372 |
| race | Asian-Pac-Islander | 480 | 133 | 0.2354 | 0.6316 | 0.0836 | 0.5470 | 0.7088 |
| race | Black | 1561 | 179 | 0.0769 | 0.5028 | 0.0217 | 0.4303 | 0.5752 |
| race | Other | 135 | 25 | 0.0963 | 0.4800 | 0.0091 | 0.3003 | 0.6650 |
| race | White | 13946 | 3490 | 0.1909 | 0.5980 | 0.0550 | 0.5816 | 0.6141 |

After retraining without sex and race, the male-minus-female recall gap is 9.79 percentage points (change +0.21). Removing these fields does not remove demographic information carried by relationship, marital status, occupation or other predictors. A smaller gap on one metric would not guarantee fairness on other metrics or intersections. These checks describe the dataset's recorded binary sex and race categories, which do not capture every identity or subgroup.

## Failure case and limits

A reproducibly selected wrong prediction, including its exact contributions, appears in [the explanation report](explainability/README.md). The selection rule is the first erroneous test record in file order. Do not tune the model to fix that record and reuse these same test scores as independent evidence.

Adult reflects historical income patterns, limited measured features and a selected 1994 sample. These models predict associations, not ability, worth, deserved pay or causal effects. Unweighted record-level results do not estimate present-day population performance. [Ding et al.](https://arxiv.org/abs/2108.04884) identify limitations to Adult's external validity; contemporary use would require contemporary data and a specified decision context.
