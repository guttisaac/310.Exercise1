"""Adult income prediction: models, explanations, and error checks."""

from pathlib import Path
import argparse
import hashlib
import html
import json
import platform
import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]
NUMERIC_FEATURES = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
CATEGORICAL_FEATURES = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(root):
    """Read both files and make their income labels consistent."""
    frames = []
    for name in ["adult.data", "adult.test"]:
        frame = pd.read_csv(
            root / "DSDA310" / name,
            names=COLUMNS,
            header=None,
            skipinitialspace=True,
            na_values="?",
            comment="|",
        )
        for column in frame.select_dtypes(include="object"):
            frame[column] = frame[column].str.strip()
        frame["income"] = frame.income.str.rstrip(".")
        assert set(frame.income.unique()) == {"<=50K", ">50K"}
        frames.append(frame)
    assert [len(f) for f in frames] == [32561, 16281]
    return frames


def make_pipeline(kind, drop=(), depth=20):
    """Put cleaning and the model together so cleaning learns from training data only."""
    numeric_features = [c for c in NUMERIC_FEATURES if c not in drop]
    categorical_features = [c for c in CATEGORICAL_FEATURES if c not in drop]
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        sparse_threshold=0,
    )
    if kind == "logistic":
        model = LogisticRegression(max_iter=3000, random_state=SEED)
    elif kind == "majority":
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=depth,
            min_samples_leaf=3,
            random_state=SEED,
            n_jobs=2,
        )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def metrics(actual, probabilities):
    """Score predictions using >50K as the positive class."""
    predicted = np.asarray(probabilities) >= 0.5
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    assert int(tn + fp + fn + tp) == len(actual)
    return dict(
        n=len(actual),
        accuracy=accuracy_score(actual, predicted),
        precision=precision_score(actual, predicted, zero_division=0),
        recall=recall_score(actual, predicted, zero_division=0),
        f1=f1_score(actual, predicted, zero_division=0),
        roc_auc=roc_auc_score(actual, probabilities),
        average_precision=average_precision_score(actual, probabilities),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def threshold_f1(estimator, X, y):
    return f1_score(y, estimator.predict_proba(X)[:, 1] >= 0.5, zero_division=0)


def md_table(frame):
    """Turn a result table into Markdown for the reports."""

    def format_value(value):
        if isinstance(value, (float, np.floating)):
            return "NA" if np.isnan(value) else f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(format_value(value) for value in values) + " |")
    return "\n".join([header, separator] + rows)


def wilson(success, total):
    """Calculate a 95% Wilson interval for a proportion."""
    if total == 0:
        return (np.nan, np.nan)
    z_score = 1.96
    proportion = success / total
    denominator = 1 + z_score * z_score / total
    center = (proportion + z_score * z_score / (2 * total)) / denominator
    margin = (
        z_score
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z_score * z_score / (4 * total * total)
        )
        / denominator
    )
    return (center - margin, center + margin)


def group_metrics(frame, actual, probability, model_name):
    """Compare error rates for the sex and race groups in the test data."""
    rows = []
    predicted = probability >= 0.5
    for field in ["sex", "race"]:
        for group in sorted(frame[field].unique()):
            group_mask = frame[field].eq(group).to_numpy()
            tn, fp, fn, tp = confusion_matrix(
                actual[group_mask], predicted[group_mask], labels=[0, 1]
            ).ravel()
            lower_bound, upper_bound = wilson(tp, tp + fn)
            rows.append(
                dict(
                    model=model_name,
                    attribute=field,
                    group=group,
                    n=int(group_mask.sum()),
                    positives=int(tp + fn),
                    base_rate=float(actual[group_mask].mean()),
                    selection_rate=float(predicted[group_mask].mean()),
                    precision=tp / (tp + fp) if tp + fp else np.nan,
                    recall=tp / (tp + fn) if tp + fn else np.nan,
                    fpr=fp / (fp + tn) if fp + tn else np.nan,
                    recall_ci_low=lower_bound,
                    recall_ci_high=upper_bound,
                    tn=int(tn),
                    fp=int(fp),
                    fn=int(fn),
                    tp=int(tp),
                )
            )
    return pd.DataFrame(rows)


def local_contributions(pipeline, row):
    """Follow each tree to see which features moved this prediction up or down."""
    preprocessor = pipeline.named_steps["preprocess"]
    forest = pipeline.named_steps["model"]
    encoded_row = preprocessor.transform(row)[0].astype(np.float32)
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][2]
    categories = (
        preprocessor.named_transformers_["cat"].named_steps["encode"].categories_
    )
    original_features = list(numeric_features) + [
        column
        for column, values in zip(categorical_features, categories)
        for _ in values
    ]
    contributions = np.zeros(len(encoded_row))
    baseline = 0.0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        node_probabilities = values[:, 1] / values.sum(axis=1)
        baseline += node_probabilities[0] / len(forest.estimators_)
        node = 0
        while tree.children_left[node] != -1:
            feature = tree.feature[node]
            child = (
                tree.children_left[node]
                if encoded_row[feature] <= tree.threshold[node]
                else tree.children_right[node]
            )
            contributions[feature] += (
                node_probabilities[child] - node_probabilities[node]
            ) / len(forest.estimators_)
            node = child
    probability = float(pipeline.predict_proba(row)[0, 1])
    assert np.isclose(baseline + contributions.sum(), probability, atol=1e-10)
    grouped = (
        pd.DataFrame({"feature": original_features, "contribution": contributions})
        .groupby("feature", as_index=False)
        .sum()
    )
    grouped["value"] = grouped.feature.map(row.iloc[0].to_dict())
    grouped = grouped.iloc[
        np.argsort(-np.abs(grouped.contribution.to_numpy()))
    ].reset_index(drop=True)
    return (baseline, probability, grouped)


def bar_svg(frame, value, path, title, subtitle):
    """Save a small bar chart without adding a plotting dependency."""
    rows = frame.head(12)
    width = 1000
    height = 150 + len(rows) * 38
    values = rows[value].to_numpy()
    largest_value = max(np.max(np.abs(values)), 1e-08)
    signed = (values < 0).any()
    origin = 620 if signed else 285
    span = 280 if signed else 570
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{height * 0.76}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="30" y="37" font-family="Arial" font-size="23" fill="#0f172a">{html.escape(title)}</text>',
        f'<text x="30" y="67" font-family="Arial" font-size="14" fill="#475569">{html.escape(subtitle)}</text>',
    ]
    for i, (_, row) in enumerate(rows.iterrows()):
        y = 98 + 38 * i
        bar_value = float(row[value])
        length = abs(bar_value) / largest_value * span
        left = origin - length if bar_value < 0 else origin
        color = "#b45309" if bar_value < 0 else "#0f766e"
        parts.extend(
            [
                f'<text x="30" y="{y + 19}" font-family="Arial" font-size="16">{html.escape(str(row.feature))}</text>',
                f'<rect x="{left}" y="{y}" width="{length}" height="25" fill="{color}"/>',
                f'<text x="925" y="{y + 19}" text-anchor="end" font-family="Arial" font-size="14">{bar_value:+.4f}</text>',
            ]
        )
    parts.append(
        f'<line x1="{origin}" y1="90" x2="{origin}" y2="{height - 40}" stroke="#64748b"/></svg>'
    )
    path.write_text("\n".join(parts))


def run(root=None):
    """Train the models, compare them, and save the project results."""
    root = Path(root or Path(__file__).resolve().parent)
    output_dir = root / "outputs"
    explanation_dir = root / "explainability"
    output_dir.mkdir(exist_ok=True)
    explanation_dir.mkdir(exist_ok=True)
    development_data, test_data = load_data(root)
    features = development_data[FEATURES]
    target = development_data.income.eq(">50K").astype(int)
    test_features = test_data[FEATURES]
    test_target = test_data.income.eq(">50K").astype(int).to_numpy()
    # Keep identical records together when making the validation split.
    fingerprints = pd.util.hash_pandas_object(
        development_data.drop(columns="income"), index=False
    )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_index, validation_index = next(
        splitter.split(features, target, groups=fingerprints)
    )
    assert not set(train_index) & set(validation_index)
    assert not set(fingerprints.iloc[train_index]) & set(
        fingerprints.iloc[validation_index]
    )
    train_features, validation_features = (
        features.iloc[train_index],
        features.iloc[validation_index],
    )
    train_target, validation_target = (
        target.iloc[train_index],
        target.iloc[validation_index],
    )
    split = pd.DataFrame(
        {"data_record": np.arange(1, len(development_data) + 1), "split": "train"}
    )
    split.loc[validation_index, "split"] = "validation"
    split.to_csv(output_dir / "development_split.csv", index=False)
    missing = pd.DataFrame(
        {"development": development_data.isna().sum(), "test": test_data.isna().sum()}
    )
    missing.to_csv(output_dir / "missing_values.csv", index_label="column")
    print(
        f"Training {len(train_index):,}; validation {len(validation_index):,}; official test {len(test_data):,}",
        flush=True,
    )
    models = {}
    scores = []
    candidates = []
    warnings.simplefilter("error", ConvergenceWarning)
    for name, kind in [
        ("Majority class", "majority"),
        ("Logistic regression", "logistic"),
    ]:
        model = make_pipeline(kind).fit(train_features, train_target)
        models[name] = model
        print(f"Fitted {name}", flush=True)
    # Choose depth using validation F1, before looking at test scores.
    for depth in [12, 20, None]:
        model = make_pipeline("forest", depth=depth).fit(train_features, train_target)
        score = metrics(
            validation_target, model.predict_proba(validation_features)[:, 1]
        )
        candidates.append((depth, score, model))
        print(f"Forest depth={depth}: validation F1={score['f1']:.4f}", flush=True)
    chosen = max(candidates, key=lambda r: r[1]["f1"])
    depth = chosen[0]
    forest = chosen[2]
    models["Random forest"] = forest
    pd.DataFrame([{"max_depth": str(d), **s} for d, s, _ in candidates]).to_csv(
        output_dir / "validation_search.csv", index=False
    )
    # Keep the chosen settings and the 0.50 threshold fixed for testing.
    test_probabilities = {}
    for name, model in models.items():
        for split_name, split_features, split_target in [
            ("train", train_features, train_target),
            ("validation", validation_features, validation_target),
            ("test", test_features, test_target),
        ]:
            probabilities = model.predict_proba(split_features)[:, 1]
            scores.append(
                {
                    "model": name,
                    "split": split_name,
                    **metrics(split_target, probabilities),
                }
            )
            if split_name == "test":
                test_probabilities[name] = probabilities
    model_scores = pd.DataFrame(scores)
    model_scores.to_csv(output_dir / "model_metrics.csv", index=False)
    prediction_table = pd.DataFrame(
        {"test_record": np.arange(1, len(test_data) + 1), "true_label": test_target}
    )
    for name, probabilities in test_probabilities.items():
        prediction_table[name + "_probability"] = probabilities
        prediction_table[name + "_prediction"] = (probabilities >= 0.5).astype(int)
    prediction_table.to_csv(output_dir / "test_predictions.csv", index=False)
    print("Comparison and test predictions saved.", flush=True)
    # Remove one set of features at a time and retrain on the same records.
    ablation = []
    variants = {}
    for name, drop in [
        ("Full features", ()),
        ("Without capital gains and losses", ("capital_gain", "capital_loss")),
        ("Without sex and race", ("sex", "race")),
    ]:
        model = (
            forest
            if not drop
            else make_pipeline("forest", drop=drop, depth=depth).fit(
                train_features, train_target
            )
        )
        variants[name] = model
        for split_name, split_features, split_target in [
            ("validation", validation_features, validation_target),
            ("test", test_features, test_target),
        ]:
            ablation.append(
                {
                    "model": name,
                    "split": split_name,
                    **metrics(split_target, model.predict_proba(split_features)[:, 1]),
                }
            )
        print(f"Checked {name}", flush=True)
    ablation_scores = pd.DataFrame(ablation)
    ablation_scores.to_csv(output_dir / "ablation_metrics.csv", index=False)
    groups = pd.concat(
        [
            group_metrics(
                test_data, test_target, m.predict_proba(test_features)[:, 1], name
            )
            for name, m in variants.items()
            if name != "Without capital gains and losses"
        ],
        ignore_index=True,
    )
    groups.to_csv(output_dir / "fairness_by_group.csv", index=False)
    overlap = (
        pd.util.hash_pandas_object(test_data.drop(columns="income"), index=False)
        .isin(set(fingerprints))
        .to_numpy()
    )
    sensitivity = metrics(
        test_target[~overlap], test_probabilities["Random forest"][~overlap]
    )
    pd.DataFrame(
        [
            {
                "scope": "All official test records",
                **metrics(test_target, test_probabilities["Random forest"]),
            },
            {"scope": "Exclude raw predictor overlap", **sensitivity},
        ]
    ).to_csv(output_dir / "overlap_sensitivity.csv", index=False)
    print("Computing global permutation importance on validation data...", flush=True)
    permutation_result = permutation_importance(
        forest,
        validation_features,
        validation_target,
        scoring=threshold_f1,
        n_repeats=5,
        random_state=SEED,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_f1_decrease": permutation_result.importances_mean,
            "std_f1_decrease": permutation_result.importances_std,
        }
    ).sort_values("mean_f1_decrease", ascending=False)
    importance.to_csv(explanation_dir / "global_importance.csv", index=False)
    bar_svg(
        importance,
        "mean_f1_decrease",
        explanation_dir / "global_importance.svg",
        "Global random forest feature importance",
        "Mean decrease in validation F1 after shuffling one raw feature; 5 repeats",
    )
    probabilities = test_probabilities["Random forest"]
    # Use the first wrong prediction in file order as the example.
    errors = np.flatnonzero((probabilities >= 0.5) != test_target)
    assert len(errors) > 0
    failure_index = int(errors[0])
    baseline_probability, failure_probability, local_importance = local_contributions(
        forest, test_features.iloc[[failure_index]]
    )
    local_importance.to_csv(explanation_dir / "local_contributions.csv", index=False)
    bar_svg(
        local_importance,
        "contribution",
        explanation_dir / "local_contributions.svg",
        f"Why test record {failure_index + 1} was misclassified",
        "Forest tree-path contributions to P(income >50K); positive raises the predicted probability",
    )
    case = {
        "test_record": failure_index + 1,
        "physical_file_line": failure_index + 2,
        "true_label": int(test_target[failure_index]),
        "predicted_label": int(probabilities[failure_index] >= 0.5),
        "probability": failure_probability,
        "baseline_probability": baseline_probability,
        "threshold": 0.5,
        "features": test_data.iloc[failure_index]
        .where(test_data.iloc[failure_index].notna(), None)
        .to_dict(),
    }
    (explanation_dir / "failure_case.json").write_text(json.dumps(case, indent=2))
    for j in [0, len(test_data) // 2, len(test_data) - 1]:
        local_contributions(forest, test_features.iloc[[j]])
    for field in ["sex", "race"]:
        for name in groups.model.unique():
            assert groups[
                (groups.model == name) & (groups.attribute == field)
            ].n.sum() == len(test_data)
    manifest = {
        "seed": SEED,
        "threshold": 0.5,
        "forest_max_depth": depth,
        "forest_trees": 200,
        "forest_min_samples_leaf": 3,
        "training_rows": len(train_index),
        "validation_rows": len(validation_index),
        "test_rows": len(test_data),
        "overlapping_test_predictor_rows": int(overlap.sum()),
        "duplicate_development_predictor_rows": int(fingerprints.duplicated().sum()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "data_sha256": {
            n: hashlib.sha256((root / "DSDA310" / n).read_bytes()).hexdigest()
            for n in ["adult.data", "adult.test"]
        },
        "checks": [
            "Expected dimensions and normalized labels",
            "Disjoint split indices and raw predictor groups",
            "Preprocessing fitted inside training pipelines",
            "Confusion matrix totals",
            "Fairness group denominators",
            "Four local probability reconstructions",
        ],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_reports(
        root,
        model_scores,
        ablation_scores,
        groups,
        importance,
        local_importance,
        case,
        manifest,
        sensitivity,
    )
    print("All checks passed. Reports and explanation artifacts generated.", flush=True)
    return {
        "metrics": model_scores,
        "ablations": ablation_scores,
        "groups": groups,
        "importance": importance,
        "local": local_importance,
        "case": case,
        "manifest": manifest,
    }


def write_reports(
    root, scores, abl, groups, importance, local, case, manifest, sensitivity
):
    test_scores = scores[scores.split == "test"].drop(columns="split")
    rf = test_scores[test_scores.model == "Random forest"].iloc[0]
    lr = test_scores[test_scores.model == "Logistic regression"].iloc[0]
    test_abl = abl[abl.split == "test"].set_index("model")
    sex = groups[
        (groups.model == "Full features") & (groups.attribute == "sex")
    ].set_index("group")
    gap = sex.loc["Male", "recall"] - sex.loc["Female", "recall"]
    nosex = groups[
        (groups.model == "Without sex and race") & (groups.attribute == "sex")
    ].set_index("group")
    gap2 = nosex.loc["Male", "recall"] - nosex.loc["Female", "recall"]
    capital_delta = test_abl.loc["Without capital gains and losses", "f1"] - rf.f1
    (root / "results.md").write_text(f"""# Model comparison results


## Final test comparison

{md_table(test_scores[['model','n','accuracy','precision','recall','f1','roc_auc','average_precision']])}

The random forest improves F1 by {rf.f1-lr.f1:.4f} over logistic regression. Its precision is {rf.precision:.1%}, while recall is {rf.recall:.1%}. Accuracy alone hides missed higher-income records. F1 balances precision and recall for the minority positive class; it excludes true negatives, so accuracy, both error rates and probability-ranking metrics are reported alongside it. There is no specified real-world error-cost ratio, so F1 is a transparent classroom choice, not a claim of optimal business utility.

## Validation and model selection

The development file is divided into {manifest['training_rows']:,} training and {manifest['validation_rows']:,} validation records using a seeded five-fold stratified group split, taking the first fold as validation. Identical raw predictor records stay in one group. Both models use the same records and features. The supplied 16,281-row test file is evaluated only after choosing forest depth on validation F1. The forest uses 200 trees, minimum leaf size 3 and selected maximum depth {manifest['forest_max_depth'] or 'unlimited'}. The candidate depths are 12, 20 and unlimited. The threshold is fixed at 0.5, not tuned. Models are not refitted on validation data.

[Validation search](outputs/validation_search.csv) records every candidate. Imputation, numeric scaling and category vocabularies are fitted only on training records. All missing-data rows remain. No test performance is used to select the model or threshold; test failure inspection is post-evaluation analysis.

## Training and validation checks

{md_table(scores[scores.split!='test'][['model','split','n','accuracy','precision','recall','f1']])}

The train-to-validation gap indicates how much performance drops outside the fitting data. These are point estimates from one split, not proof of a statistically significant improvement. The original notebooks had already examined parts of the development data, so this is a repaired evaluation protocol rather than a preregistered experiment.

## Confusion matrices

Rows are actual labels and columns are predicted labels. In each row below, TN means actual <=50K predicted <=50K; FP means actual <=50K predicted >50K; FN means actual >50K predicted <=50K; TP means actual >50K predicted >50K.

{md_table(test_scores[['model','tn','fp','fn','tp']])}

All matrices total 16,281. [Raw predictions](outputs/test_predictions.csv) permit independent recalculation. [Run manifest](outputs/run_manifest.json) records versions, seeds, data hashes and checks.
""")
    (root / "fairness_leakage_check.md").write_text(
        f"""# Evaluation and failure analysis

This report completes the September 29 analysis scope using measured results. It was produced on October 3, 2026; it does not claim the work was completed by the milestone date.

## Leakage risk and actual removal test

The original proposal stated that no major leakage was seen; it did not name a specific risky feature. The risk below is a newly documented completion-session hypothesis, not a retroactively claimed September 3 decision.

Assumption: if income is predicted before year-end, same-year capital gains and losses may be unavailable or too closely tied to the outcome period. They are legitimate benchmark inputs when available at prediction time; their predictive value alone does not prove leakage. We retrained the same forest configuration on the same training records after removing both fields, retaining the 0.5 threshold. No replacement feature or hyperparameter was selected using test results.

{md_table(abl[['model','split','accuracy','precision','recall','f1']])}

Removing capital fields changes test F1 by {capital_delta:+.4f}. This measures dependence on the fields, not proof of their timing or causal relationship to income. If prediction-time availability cannot be established, use the reduced-input specification and evaluate it for the actual use case. The target is excluded from all feature lists. All fitted preprocessing uses training records only.

## Duplicate and overlap robustness

The development data contains {manifest['duplicate_development_predictor_rows']} repeated raw predictor rows. Grouping prevents these identical predictor records from crossing training and validation. There are {manifest['overlapping_test_predictor_rows']} official test records whose full raw predictor tuple also appears in development. We retain the supplied benchmark for the primary result and additionally score the nonoverlapping records: n={sensitivity['n']:,}, accuracy={sensitivity['accuracy']:.4f}, F1={sensitivity['f1']:.4f}. This does not establish that remaining records are independent people; there is no person identifier. See [overlap sensitivity](outputs/overlap_sensitivity.csv).

## Group performance

All rates below are measured on the same official test file. Base rate is the actual higher-income share; selection rate is the predicted higher-income share; recall is TP/(TP+FN); false-positive rate is FP/(FP+TN). Recall intervals are approximate 95% Wilson intervals for each group, not confidence intervals for between-group differences. NA means a rate has no denominator.

{md_table(groups[groups.model=='Full features'][['attribute','group','n','positives','base_rate','selection_rate','precision','recall','fpr','recall_ci_low','recall_ci_high']])}

Male recall is {sex.loc['Male','recall']:.1%} and female recall is {sex.loc['Female','recall']:.1%}, a signed male-minus-female gap of {gap*100:.2f} percentage points. This shows unequal detection of higher-income individuals in this sample. It does not establish discrimination as a causal conclusion. Race-group denominators and intervals matter: small positive counts make recall estimates unstable. No blanket fairness pass is asserted.

## Removing sex and race

{md_table(groups[groups.model=='Without sex and race'][['attribute','group','n','positives','selection_rate','recall','fpr','recall_ci_low','recall_ci_high']])}

After retraining without sex and race, the male-minus-female recall gap is {gap2*100:.2f} percentage points (change {(gap2-gap)*100:+.2f}). Removing these fields does not remove demographic information carried by relationship, marital status, occupation or other predictors. A smaller gap on one metric would not guarantee fairness on other metrics or intersections. These checks describe the dataset's recorded binary sex and race categories, which do not capture every identity or subgroup.

## Failure case and limits

A reproducibly selected wrong prediction, including its exact contributions, appears in [the explanation report](explainability/README.md). The selection rule is the first erroneous test record in file order. Do not tune the model to fix that record and reuse these same test scores as independent evidence.

Adult reflects historical income patterns, limited measured features and a selected 1994 sample. These models predict associations, not ability, worth, deserved pay or causal effects. Unweighted record-level results do not estimate present-day population performance. [Ding et al.](https://arxiv.org/abs/2108.04884) identify limitations to Adult's external validity; contemporary use would require contemporary data and a specified decision context.
"""
    )
    negatives = local.sort_values("contribution").head(3)
    positives = local.sort_values("contribution", ascending=False).head(3)
    wrongtype = "false negative" if case["true_label"] == 1 else "false positive"
    (root / "explainability" / "README.md").write_text(
        f"""# Global explanation and individual failure case

This analysis explains the selected random forest and diagnoses one actual test error. It uses validation permutation importance globally and exact forest tree-path probability contributions locally. Neither method supplies causal effects.

## Global feature importance

![Global permutation importance](global_importance.svg)

{md_table(importance)}

Each original feature is shuffled five times on validation data, leaving the model fixed. Larger mean F1 decreases indicate greater reliance under this perturbation. The leading features are {', '.join(importance.feature.head(3))}. Standard deviations describe variation across shuffles, not confidence intervals. Correlated or redundant predictors can share importance; shuffling can create implausible feature combinations. Importance gives neither causal direction nor an individual's explanation.

## One specific wrong prediction

Official adult.test data record **{case['test_record']}** (one-based, excluding the header) is a **{wrongtype}**. Its actual label is **{'above $50K' if case['true_label'] else 'at or below $50K'}**, but the model predicts **{'above $50K' if case['predicted_label'] else 'at or below $50K'}**. The predicted higher-income probability is **{case['probability']:.4f}**, compared with the fixed 0.50 threshold. This is the first error in file order, not a representative sample of every error.

{md_table(pd.DataFrame({'field':case['features'].keys(),'value':case['features'].values()}))}

## Local feature contributions

![Individual prediction explanation](local_contributions.svg)

{md_table(local)}

For each tree, follow this individual's route from root to leaf and assign each change in the positive-class fraction to the feature used at that split. Average across trees and aggregate one-hot columns back to their original features. The average root probability is **{case['baseline_probability']:.6f}**; adding all feature contributions gives **{case['probability']:.6f}**, equal to predict_proba within numerical tolerance. Positive contributions raise the model's estimate; negative contributions lower it. This is a tree-path decomposition, not SHAP, and allocations depend on tree structure and correlated features.

The strongest downward contributions are {', '.join(f"{r.feature}={r.value} ({r.contribution:+.4f})" for r in negatives.itertuples())}. The strongest upward contributions are {', '.join(f"{r.feature}={r.value} ({r.contribution:+.4f})" for r in positives.itertuples())}. Their combined balance leaves the probability on the wrong side of 0.50 despite the observed income label. This explains the model's computation; it cannot establish why this person's real income differed. Coarse education and occupation categories omit individual earnings details, and income groups overlap within observed profiles. Those are plausible limitations rather than verified facts about this individual.

A next experiment could compare error types across occupation and age, then evaluate a prespecified threshold or additional legitimately available predictors on fresh data. Lowering a threshold can reduce false negatives while increasing false positives; changing it just to correct this example would overfit the evaluation.
"""
    )
    (root / "executive_summary.md").write_text(
        f"""# Adult income prediction executive summary draft


## Question and approach

Can census characteristics distinguish people earning above $50,000 per year from those earning at or below that amount? We analyzed 48,842 historical Adult dataset records, with one record per individual. The original development file was divided into training and validation groups, while the separate 16,281-record test file measured final performance. Missing entries were filled using training-data values. We compared a simple majority-class rule, logistic regression and a random forest; model settings were chosen using validation data.

## Main findings

On the final test file, logistic regression reached {lr.accuracy:.1%} accuracy and {lr.f1:.3f} F1, while the random forest reached {rf.accuracy:.1%} accuracy and {rf.f1:.3f} F1. F1 balances correct positive predictions with detection of higher-income individuals. We emphasized it because a model can achieve high accuracy by predicting the much more common lower-income class.

For the forest, {rf.precision:.1%} of higher-income predictions were correct, and it detected {rf.recall:.1%} of actual higher-income records. It missed {int(rf.fn):,} higher-income people and incorrectly classified {int(rf.fp):,} lower-income people as higher-income. The strongest global signals were {', '.join(importance.feature.head(3)).replace('_',' ')}. An individual error analysis demonstrates that population patterns can still produce the wrong prediction for a particular person.

## Risks and recommendation

Removing capital gains and losses changed F1 by {capital_delta:+.3f}; these inputs need to be available when a prediction is made. Higher-income recall differed by {gap*100:.1f} percentage points between male and female records. Removing sex and race did not eliminate the need to examine group errors because other features can convey related information. Smaller race groups have less precise estimates.

Use this as an educational comparison of predictive methods. The random forest improves F1 in this evaluation, but historical data, incomplete personal information, unequal group errors and uncertain prediction-time availability limit practical use. Before any real application, define the decision and error costs, obtain suitable current data, and repeat evaluation. These results do not measure a person's ability or deserved pay. AI helped implement and document the analysis; the team should review the evidence and be able to explain the choices before submission.
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    run(parser.parse_args().root)
