import numpy as np

from supervised_baseline import load_data, score, FEATURES


def test_load_data_returns_the_shared_feature_columns(tiny_csv):
    X, y = load_data(tiny_csv)
    assert list(X.columns) == FEATURES
    assert set(y.unique()) <= {0, 1}


def test_score_computes_expected_confusion_matrix_and_metrics():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])  # one false negative
    m = score("test", y_true, y_pred)

    assert m["accuracy"] == 0.75
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 2
    assert m["fp"] == 0
    assert m["precision"] == 1.0   # no false positives
    assert m["recall"] == 0.5      # caught 1 of 2 actual positives


def test_score_handles_no_predicted_positives_without_error():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0, 0, 0, 0])
    m = score("test", y_true, y_pred)
    # zero_division=0 should keep this from raising or returning NaN
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0
