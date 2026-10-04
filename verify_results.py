
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from analysis import load_data, threshold_f1

ROOT = Path(__file__).resolve().parent


def verify(root=ROOT):
    root = Path(root)
    predictions = pd.read_csv(root / "outputs/test_predictions.csv")
    scores = pd.read_csv(root / "outputs/model_metrics.csv")
    groups = pd.read_csv(root / "outputs/fairness_by_group.csv")
    dev, test = load_data(root)
    y = predictions.true_label.to_numpy()
    assert np.array_equal(y, test.income.eq(">50K").astype(int))
    assert len(predictions) == 16281
    for row in scores[scores.split == "test"].itertuples():
        p = predictions[row.model + "_probability"].to_numpy()
        pred = p >= 0.5
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        assert np.array_equal(pred, predictions[row.model + "_prediction"])
        for key, fn in [
            ("accuracy", accuracy_score),
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ]:
            actual = fn(y, pred) if key == "accuracy" else fn(y, pred, zero_division=0)
            assert np.isclose(actual, getattr(row, key))
        assert tuple(confusion_matrix(y, pred, labels=[0, 1]).ravel()) == (
            row.tn,
            row.fp,
            row.fn,
            row.tp,
        )
    forest = predictions["Random forest_prediction"].to_numpy()
    for row in groups[groups.model == "Full features"].itertuples():
        mask = test[row.attribute].eq(row.group).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y[mask], forest[mask], labels=[0, 1]).ravel()
        assert (tn, fp, fn, tp) == (row.tn, row.fp, row.fn, row.tp)
        assert row.n == mask.sum() and row.positives == tp + fn
        assert np.isclose(row.recall, tp / (tp + fn))
        assert np.isclose(row.fpr, fp / (fp + tn))
    split = pd.read_csv(root / "outputs/development_split.csv")
    assert np.array_equal(split.data_record, np.arange(1, len(dev) + 1))
    keys = pd.util.hash_pandas_object(dev.drop(columns="income"), index=False)
    assert not set(keys[split.split.eq("train")]) & set(
        keys[split.split.eq("validation")]
    )
    case = json.loads((root / "explainability/failure_case.json").read_text())
    local = pd.read_csv(root / "explainability/local_contributions.csv")
    assert np.isclose(
        case["baseline_probability"] + local.contribution.sum(), case["probability"]
    )
    i = case["test_record"] - 1
    assert case["true_label"] == int(y[i]) and case["predicted_label"] != int(y[i])
    assert np.isclose(
        case["probability"], predictions["Random forest_probability"].iloc[i]
    )
    tied = DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1])
    assert np.array_equal(tied.predict([[0], [1]]), [0, 0])
    assert np.isclose(threshold_f1(tied, [[0], [1]], [0, 1]), 2 / 3)
    print(
        "PASS: exported metrics, confusion matrices, group rates, split separation, failure-case reconstruction and threshold-tie consistency."
    )


if __name__ == "__main__":
    verify()
