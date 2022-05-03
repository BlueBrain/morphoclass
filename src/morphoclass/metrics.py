# Copyright © 2022-2022 Blue Brain Project/EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of various metric functions."""
from __future__ import annotations

import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
from sklearn.metrics._classification import _check_targets


def chance_accuracy(y_true: ArrayLike[int]) -> float:
    """Compute accuracy obtained "by chance" by a model giving random predictions.

    Parameters
    ----------
    y_true
        1d array-like. Ground truth (correct) labels.

    Returns
    -------
    score : float
        The expected accuracy of a model giving random predictions according to the
        class frequencies observed in `y_true`.

    Raises
    ------
    ValueError
        If `y_true` has no elements.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Cohen%27s_kappa#Definition
    [2] https://github.com/BlueBrain/morphoclass/issues/49#issue-1224044038
    """
    if not y_true:
        raise ValueError(f"y_true = {y_true} is empty!")
    _check_targets(y_true, y_true)
    _, counts = np.unique(y_true, return_counts=True)
    return (counts**2).sum()


def inter_rater_score(targets, pred, kind="cohen_kappa"):
    """Compute different kinds of inter-rater agreement scores.

    Also known as the "chance corrected accuracy".

    The different kinds are either "Cohen Kappa" or "Scott's Pi".

    The score is equal to zero if the agreement p0 is purely due to
    the expected chance agreement pe. The difference between the
    Cohen Kappa score and the Scott's Pi score is that the former
    computes the chance agreement as squared geometric means of
    marginal proportions, and the latter as squared arithmetic means.

    Cohen's kappa is considered more informative since Scott's Pi
    since it assumes that the annotators draw their responses from
    different distributions, while Scott's Pi doesn't.

    Parameters
    ----------
    targets
        True labels.
    pred
        Predicted labels.
    kind
        Kind of score to be computed. Either `"cohen_kappa"` or `"scotts_pi"`.

    Returns
    -------
    score
        The inter-rater score.
    """
    confusion = metrics.confusion_matrix(targets, pred)
    tot = confusion.sum()
    p0 = confusion.diagonal().sum() / tot
    pv = confusion.sum(axis=0)
    ph = confusion.sum(axis=1)

    if kind == "cohen_kappa":
        # Geometric joint proportions
        pe = np.vdot(pv, ph) / (tot * tot)
    elif kind == "scotts_pi":
        # Arithmetic joint proportions
        pe = np.linalg.norm((pv + ph) / (tot + tot)) ** 2
    else:
        raise ValueError(f"Score kind '{kind}' not valid")

    return (p0 - pe) / (1 - pe)


def discordance(targets, probabilities):
    """Compute the discordance score given binary classification probabilities.

    Parameters
    ----------
    targets
        Array with true binary labels.
    probabilities
        Array with prediction probabilities for binary classification.

    Returns
    -------
    score
        The discordance score.
    """
    t = np.array(targets)
    p = np.array(probabilities)

    assert set(np.unique(targets)).issubset({0, 1}), "Targets are non-binary"
    assert t.shape == p.shape, "Targets and probability arrays have different lengths"

    len1 = np.sum(t == 1)
    len0 = np.sum(t == 0)

    # Ascending order of the probabilities where in cas of ties
    # target=0 is placed before target=1, resulting in a more forgiving
    # score. Change 'zip(t, p)' to 'zip(-t, p)' below for the reverse,
    # and more punishing handling of ties.
    # In a realistic setting ties should occur only rarely, but the handling
    # of ties is nonetheless useful for consistency of results.
    tp = np.array(list(zip(t, p)), dtype=[("t", np.int32), ("p", np.float32)])
    ordered = np.argsort(tp, order=["p", "t"])

    count = 0
    inc = 0
    for x in t[ordered]:
        if x:
            inc += 1
        else:
            count += inc

    return count / (len0 * len1)


def discordance_multiclass(targets, probabilities):
    """Compute the discordance score.

    Parameters
    ----------
    targets
        Array with true (sparse) multi-class labels of shape `(n_samples,)`.
    probabilities
        Array with multi-class classification probabilities of shape
        `(n_samples, n_classes)`.

    Returns
    -------
    results
        Array of shape `(n_classes)` containing discordance scores for each
        class computed as a one-versus-all binary classification.
    """
    results = []
    for cls in range(probabilities.shape[-1]):
        cls_targets = np.array(targets == cls, dtype="int")
        cls_probabilities = probabilities[:, cls]
        results.append(discordance(cls_targets, cls_probabilities))
    return np.array(results)


def fowlkes_mallows_score(targets, pred):
    """Compute the Fowlkes-Mallows score a.k.a. G-means.

    It is similar to the F1 score, but instead of a harmonic mean of the
    precision and recall, a geometric mean is taken.

    Parameters
    ----------
    targets
        True labels.
    pred
        Predicted labels.

    Returns
    -------
    score
        The Fowlkes-Mallows score
    """
    precision = metrics.precision_score(targets, pred, average=None)
    recall = metrics.recall_score(targets, pred, average=None)
    score = np.sqrt(precision * recall)

    return score


def classification_report(targets, pred_proba, labels=None):
    """Create an advanced classification report.

    targets
        The target labels.
    pred_proba
        Predicted label probabilities.
    labels
        Descriptions for the numerical labels.
    """
    t = np.array(targets)
    pred = pred_proba.argmax(axis=1)
    if t.shape != pred.shape:
        raise ValueError("Shapes of targets and predictions not compatible")

    if labels is None:
        labels = ["" for _ in range(pred_proba.shape[1])]
    else:
        max_len = max(len(label) for label in labels)
        labels = [f"(= {label.ljust(max_len)})" for label in labels]

    print("Class distributions:")
    counts = np.unique(targets, return_counts=True)
    for cls, n in zip(*counts):
        print(f"{cls} {labels[cls]}: {n:3d}")
    print("-" * 24)
    print(f"Total : {len(targets):3d}")
    fig = Figure()
    ax = fig.subplots()
    ax.bar(*counts, tick_label=counts[0])
    ax.set_title("Class distributions")
    ax.set_xlabel("Class")
    ax.set_ylabel("Counts")
    # TODO: either delete, save or return figure
    # plt.show()

    fig = Figure()
    ax = fig.subplots()
    sns.heatmap(metrics.confusion_matrix(targets, pred), annot=True, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    # TODO: either delete, save or return figure
    # plt.show()

    report = metrics.classification_report(targets, pred)

    accuracy = metrics.accuracy_score(targets, pred)
    balanced_accuracy = metrics.balanced_accuracy_score(targets, pred)

    matthews = metrics.matthews_corrcoef(targets, pred)
    cohen_kappa = inter_rater_score(targets, pred, kind="cohen_kappa")
    scotts_pi = inter_rater_score(targets, pred, kind="scotts_pi")

    f1_scores = metrics.f1_score(targets, pred, average=None)
    fowlkes_mallows = fowlkes_mallows_score(targets, pred)
    discordance_scores = discordance_multiclass(targets, pred_proba)

    print(report)
    print(f"Accuracy:\t {accuracy:.2f}")
    print(f"Balanced acc.:\t {balanced_accuracy:.2f}")
    print()

    print(f"Matthew's Coef:\t {matthews:.2f}")
    print(f"Cohen Kappa:\t {cohen_kappa:.2f}")
    print(f"Scott's Pi:\t {scotts_pi:.2f}")
    print()

    np.set_printoptions(precision=2)
    print("Classes:\t " + "\t".join(f"{i:4d}" for i in range(pred_proba.shape[1])))
    print("F1:\t\t " + "\t".join(f"{x:.2f}" for x in f1_scores))
    print("Fowlkes-Mallows: " + "\t".join(f"{x:.2f}" for x in fowlkes_mallows))
    print("Discordance:\t " + "\t".join(f"{x:.2f}" for x in discordance_scores))
    np.set_printoptions()
