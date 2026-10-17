import numpy as np
from shared.metrics import summarize


def test_metrics_are_correct():
    truth = [0, 0, 1, 1]
    pred = [0, 1, 1, 1]
    prob = np.array([[.9, .1], [.3, .7], [.1, .9], [.2, .8]])
    scores, table, cm, calibration = summarize(truth, pred, prob, n_classes=2)
    assert scores["accuracy"] == .75
    assert scores["balanced_accuracy"] == .75
    assert cm.tolist() == [[1, 1], [0, 2]]
    assert abs(scores["macro_f1"] - (2/3 + 4/5)/2) < 1e-7
    assert len(table) == 2
    assert calibration["count"].sum() == 4
