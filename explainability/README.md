# Global explanation and individual failure case

This analysis explains the selected random forest and diagnoses one actual test error. It uses validation permutation importance globally and exact forest tree-path probability contributions locally. Neither method supplies causal effects.

## Global feature importance


| feature | mean_f1_decrease | std_f1_decrease |
| --- | --- | --- |
| marital_status | 0.1199 | 0.0067 |
| relationship | 0.1134 | 0.0053 |
| education_num | 0.0922 | 0.0062 |
| capital_gain | 0.0917 | 0.0028 |
| occupation | 0.0615 | 0.0018 |
| age | 0.0518 | 0.0073 |
| hours_per_week | 0.0315 | 0.0030 |
| capital_loss | 0.0227 | 0.0018 |
| sex | 0.0198 | 0.0037 |
| workclass | 0.0143 | 0.0031 |
| native_country | 0.0024 | 0.0019 |
| race | 0.0015 | 0.0009 |

Each original feature is shuffled five times on validation data, leaving the model fixed. Larger mean F1 decreases indicate greater reliance under this perturbation. The leading features are marital_status, relationship, education_num. Standard deviations describe variation across shuffles, not confidence intervals. Correlated or redundant predictors can share importance; shuffling can create implausible feature combinations. Importance gives neither causal direction nor an individual's explanation.

## One specific wrong prediction

Official adult.test data record **3** (one-based, excluding the header) is a **false negative**. Its actual label is **above $50K**, but the model predicts **at or below $50K**. The predicted higher-income probability is **0.4036**, compared with the fixed 0.50 threshold. This is the first error in file order, not a representative sample of every error.

| field | value |
| --- | --- |
| age | 28 |
| workclass | Local-gov |
| fnlwgt | 336951 |
| education | Assoc-acdm |
| education_num | 12 |
| marital_status | Married-civ-spouse |
| occupation | Protective-serv |
| relationship | Husband |
| race | White |
| sex | Male |
| capital_gain | 0 |
| capital_loss | 0 |
| hours_per_week | 40 |
| native_country | United-States |
| income | >50K |

## Local feature contributions


| feature | contribution | value |
| --- | --- | --- |
| age | -0.1084 | 28 |
| marital_status | 0.0979 | Married-civ-spouse |
| relationship | 0.0730 | Husband |
| workclass | 0.0695 | Local-gov |
| occupation | 0.0518 | Protective-serv |
| capital_gain | -0.0246 | 0 |
| hours_per_week | -0.0170 | 40 |
| sex | 0.0143 | Male |
| education_num | 0.0117 | 12 |
| capital_loss | -0.0090 | 0 |
| native_country | 0.0021 | United-States |
| race | 0.0014 | White |

For each tree, follow this individual's route from root to leaf and assign each change in the positive-class fraction to the feature used at that split. Average across trees and aggregate one-hot columns back to their original features. The average root probability is **0.240945**; adding all feature contributions gives **0.403592**, equal to predict_proba within numerical tolerance. Positive contributions raise the model's estimate; negative contributions lower it. This is a tree-path decomposition, not SHAP, and allocations depend on tree structure and correlated features.

The strongest downward contributions are age=28 (-0.1084), capital_gain=0 (-0.0246), hours_per_week=40 (-0.0170). The strongest upward contributions are marital_status=Married-civ-spouse (+0.0979), relationship=Husband (+0.0730), workclass=Local-gov (+0.0695). Their combined balance leaves the probability on the wrong side of 0.50 despite the observed income label. This explains the model's computation; it cannot establish why this person's real income differed. Coarse education and occupation categories omit individual earnings details, and income groups overlap within observed profiles. Those are plausible limitations rather than verified facts about this individual.

A next experiment could compare error types across occupation and age, then evaluate a prespecified threshold or additional legitimately available predictors on fresh data. Lowering a threshold can reduce false negatives while increasing false positives; changing it just to correct this example would overfit the evaluation.
