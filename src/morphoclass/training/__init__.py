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
"""Utilities for setting up and running model training."""
from __future__ import annotations

from morphoclass.training._helpers import create_k_folds
from morphoclass.training._helpers import reset_seeds
from morphoclass.training.rd_dataset_prep import make_transform
from morphoclass.training.rd_dataset_prep import prepare_rd_split
from morphoclass.training.rd_dataset_prep import prepare_rd_transforms
from morphoclass.training.rd_dataset_prep import prepare_smart_split
from morphoclass.training.reports.transfer_learning_report import (
    transfer_learning_report,
)
from morphoclass.training.training_loop_cv import train_model_cv
from morphoclass.training.training_loop_loo import train_model_loo
from morphoclass.training.training_loop_regression import train_regression_model
from morphoclass.training.training_loop_simple import train_model
from morphoclass.training.transfer_learning import transfer_learning_curves

# from morphoclass.training.tns_utils import (
#     read_tns_parameters,
#     tns_distributions_from_dataset,
# )

__all__ = [
    "reset_seeds",
    "create_k_folds",
    "train_model",
    "train_model_cv",
    "train_model_loo",
    "train_regression_model",
    "make_transform",
    "prepare_rd_split",
    "prepare_rd_transforms",
    "prepare_smart_split",
    "transfer_learning_curves",
    "transfer_learning_report",
    # "tns_distributions_from_dataset",
    # "read_tns_parameters",
]
