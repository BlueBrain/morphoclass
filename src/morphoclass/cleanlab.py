# Copyright © 2022 Blue Brain Project/EPFL
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
"""CleanLab analysis of labeled data."""
from __future__ import annotations

import warnings

import numpy as np


def check_installed() -> bool:
    """Check whether the CleanLab package is installed.

    Returns
    -------
    bool
        Whether the CleanLab package is installed
    """
    try:
        import cleanlab  # noqa: F401
    except ImportError:
        return False
    else:
        return True


def how_to_install_msg() -> str:
    """Get installation instructions for CleanLab.

    Returns
    -------
    str
        The instructions on how to install CleanLab.
    """
    return (
        "To install the version <version> of CleanLab run "
        '"pip install cleanlab==<version>". morphoclass was tested using '
        'CleanLab version "1.0". To install the latest version of CleanLab '
        'run "pip install cleanlab".'
    )


def warn_if_not_installed() -> None:
    """Issue a UserWarning if CleanLab is not installed."""
    if not check_installed():
        warnings.warn(
            f"CleanLab is not installed. {how_to_install_msg()}",
            stacklevel=3,
        )


def analysis(labels: np.ndarray, probas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use CleanLab to detect potential label errors.

    Parameters
    ----------
    labels
        The labels of the data. Should have shape (n_samples,).
    probas
        The class probabilities predicted by a trained model. Should have
        shape (n_samples, n_classes).

    Returns
    -------
    label_error_ids : np.ndarray
        The indices of data with potential label errors found by CleanLab.
        The shape is (n_bad_labels,).
    self_confidence : np.ndarray
        The self-confidence score of the outliers, shape is (n_bad_labels,).
        The self-confidence is the predicted probability of the true label.
    """
    from cleanlab.pruning import get_noise_indices

    label_error_ids = get_noise_indices(
        s=labels,
        psx=probas,
        sorted_index_method="normalized_margin",  # Orders label errors
    )
    self_confidence = np.array([probas[idx][labels[idx]] for idx in label_error_ids])

    return label_error_ids, self_confidence
