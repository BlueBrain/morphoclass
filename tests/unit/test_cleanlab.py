from __future__ import annotations

import sys

import numpy as np
import pytest

from morphoclass import cleanlab

pytest.importorskip("cleanlab")


def test_check_installed(monkeypatch):
    assert cleanlab.check_installed() is True

    monkeypatch.setitem(sys.modules, "cleanlab", None)
    assert cleanlab.check_installed() is False


def test_how_to_install_msg():
    msg = cleanlab.how_to_install_msg()
    assert "pip install cleanlab" in msg


class TestAnalysis:
    def test_no_errors(self):
        probas = np.array([[0.99, 0.01], [0.15, 0.85]])
        labels = np.array([0, 1])
        error_ids, confidence = cleanlab.analysis(labels, probas)
        assert len(error_ids) == 0
        assert len(confidence) == 0

    def test_with_errors(self):
        probas = np.array([[0.99, 0.01], [0.15, 0.85]] * 100)
        labels = np.array([0, 1] * 100)
        labels[5] = 1 - labels[5]
        labels[33] = 1 - labels[33]
        labels[80] = 1 - labels[80]
        error_ids, confidence = cleanlab.analysis(labels, probas)
        assert sorted(error_ids) == [5, 33, 80]
        assert np.array_equal(confidence, probas[error_ids, labels[error_ids]])
