from __future__ import annotations

import numpy as np
import pytest

import morphoclass.metrics


@pytest.fixture
def targets():
    return [0, 1, 2, 3, 0, 1, 2, 3]


@pytest.fixture
def predictions():
    return [0, 1, 2, 3, 0, 0, 0, 0]


@pytest.fixture
def targets_long():
    return np.array(
        [
            1,
            0,
            2,
            1,
            0,
            3,
            1,
            3,
            2,
            1,
            0,
            0,
            2,
            1,
            0,
            0,
            2,
            0,
            0,
            2,
            0,
            1,
            0,
            2,
            0,
            3,
            0,
            3,
            1,
            1,
            1,
            2,
            1,
            0,
            3,
            0,
            2,
            3,
            3,
            0,
        ]
    )


@pytest.fixture
def predictions_proba_long():
    return np.array(
        [
            [2.13409236e-04, 9.99786556e-01, 3.58841315e-15, 7.21092932e-22],
            [4.00791794e-01, 5.84340811e-01, 1.48593979e-02, 7.94825701e-06],
            [6.05249882e-01, 8.68840609e-03, 2.52534956e-01, 1.33526787e-01],
            [1.54491485e-04, 9.99845505e-01, 3.75759710e-12, 2.06046463e-25],
            [1.05908187e-02, 9.89193380e-01, 2.15820372e-04, 5.94677571e-11],
            [3.94736358e-04, 2.37939779e-10, 5.63643081e-03, 9.93968844e-01],
            [9.87039149e-01, 1.29607795e-02, 4.98485855e-13, 1.52073642e-21],
            [2.08745114e-04, 5.57047564e-08, 2.28072524e-01, 7.71718681e-01],
            [8.32691967e-01, 9.96807939e-05, 1.61802813e-01, 5.40559879e-03],
            [8.32431078e-01, 1.47444949e-01, 1.21262698e-02, 7.99777266e-03],
            [1.19035028e-01, 8.80813122e-01, 1.51805973e-04, 2.48517793e-08],
            [9.98500228e-01, 1.49964180e-03, 1.76246900e-07, 6.52209811e-16],
            [8.43541929e-05, 4.74865125e-08, 9.92079318e-01, 7.83621799e-03],
            [8.14772606e-01, 1.65875047e-01, 1.93516649e-02, 6.67125221e-07],
            [9.99780715e-01, 1.26418036e-05, 1.74102370e-05, 1.89278260e-04],
            [9.36312914e-01, 6.36865199e-02, 6.04160618e-07, 1.37202778e-15],
            [4.71147627e-01, 2.20356379e-02, 5.06816745e-01, 1.34950464e-15],
            [9.97773230e-01, 2.22500949e-03, 1.82203303e-06, 1.04441278e-09],
            [9.90413964e-01, 6.04134193e-03, 3.53308185e-03, 1.16337069e-05],
            [1.93873689e-06, 1.29684409e-07, 9.99997973e-01, 8.49136247e-11],
            [9.43612456e-01, 5.63872382e-02, 3.86100879e-07, 3.57663965e-09],
            [1.08756019e-07, 9.99999881e-01, 4.56874447e-13, 4.95637229e-29],
            [4.97953832e-01, 5.01565635e-01, 4.78504808e-04, 1.98120438e-06],
            [2.20465037e-04, 6.43653841e-09, 5.95636293e-02, 9.40215886e-01],
            [9.52643573e-01, 4.73561548e-02, 2.31179740e-07, 4.30250384e-11],
            [1.12411240e-03, 5.22170609e-08, 1.35102004e-01, 8.63773823e-01],
            [8.01629424e-01, 1.93965524e-01, 4.40499606e-03, 3.64478403e-08],
            [3.09047289e-04, 1.42487744e-09, 5.25377598e-03, 9.94437218e-01],
            [7.14591442e-05, 9.99928594e-01, 1.46118886e-08, 5.97088579e-18],
            [1.38697838e-02, 9.86128807e-01, 1.40058398e-06, 5.57964108e-16],
            [7.64748514e-01, 2.35251456e-01, 8.89148311e-08, 2.53417860e-12],
            [1.58699438e-01, 5.24707092e-03, 6.97396219e-01, 1.38657287e-01],
            [4.57621008e-01, 5.39143920e-01, 3.23496037e-03, 1.13475714e-07],
            [9.99964833e-01, 3.48067515e-05, 3.53509989e-07, 2.04100168e-08],
            [5.30223250e-02, 3.37367442e-06, 1.55043835e-02, 9.31469977e-01],
            [9.99964595e-01, 3.53475189e-05, 1.41902684e-10, 5.94006224e-12],
            [2.67553260e-03, 2.27736782e-05, 4.50574607e-01, 5.46727121e-01],
            [5.54893445e-03, 1.37592465e-07, 1.07413791e-02, 9.83709574e-01],
            [9.37969759e-02, 1.13401211e-05, 1.17022609e-02, 8.94489467e-01],
            [9.83091295e-01, 1.69013068e-02, 7.41942722e-06, 1.60727876e-12],
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize(
    "kind,score", [("cohen_kappa", 0.5), ("scotts_pi", 0.466666666666666)]
)
def test_inter_rater_score(targets, predictions, kind, score):
    assert np.allclose(
        morphoclass.metrics.inter_rater_score(targets, predictions, kind=kind),
        score,
    )


def test_inter_rater_score_fail(targets, predictions):
    with pytest.raises(Exception):
        morphoclass.metrics.inter_rater_score(targets, predictions, kind="invalid")


def test_discordance():
    targets = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    probabilities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.4, 0.3, 0.2]
    assert morphoclass.metrics.discordance(targets, probabilities) == 0.1


def test_discordance_multiclass(targets_long, predictions_proba_long):
    expected = np.array([0.11466667, 0.09333333, 0.01171875, 0.01731602])
    result = morphoclass.metrics.discordance_multiclass(
        targets_long, predictions_proba_long
    )
    assert np.allclose(result, expected)


def test_fowlkes_mallows_score(targets, predictions):
    expected = [0.63245553, 0.70710678, 0.70710678, 0.70710678]
    result = morphoclass.metrics.fowlkes_mallows_score(targets, predictions)
    assert np.allclose(result, expected)


@pytest.mark.parametrize("labels", [None, ["a", "b", "c", "d"]])
def test_classification_report(labels, targets_long, predictions_proba_long):
    morphoclass.metrics.classification_report(
        targets_long, predictions_proba_long, labels=labels
    )
