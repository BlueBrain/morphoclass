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
"""Implementation of the training config."""
from __future__ import annotations

import dataclasses
import importlib
import pathlib
from typing import Callable

import yaml


@dataclasses.dataclass
class TrainingConfig:
    """A training configuration."""

    input_csv: pathlib.Path | None
    model_class: str
    model_params: dict
    splitter_class: str
    splitter_params: dict
    dataset_name: str
    feature_extractor_name: str
    optimizer_class: str
    optimizer_params: dict
    n_epochs: int
    batch_size: int
    seed: int
    oversampling: bool = False  # This used to be set by the dataset config
    neurite_type: str | None = None  # if None then fall back to default of dataset type
    train_all_samples: bool = False
    checkpoint_path_pretrained: pathlib.Path | None = None
    frozen_backbone: bool = False

    def __post_init__(self):
        """Run post-initialisation steps."""
        if self.input_csv:
            self.input_csv = pathlib.Path(self.input_csv)

    @staticmethod
    def import_obj(obj_full_name: str) -> object:
        """Import an object given its full module and class path."""
        module_name, _, obj_name = obj_full_name.rpartition(".")
        module = importlib.import_module(module_name)
        obj = getattr(module, obj_name)
        return obj

    @property
    def model_cls(self) -> Callable:
        """Get the model class."""
        model_cls = self.import_obj(self.model_class)
        if not callable(model_cls):
            raise ValueError(f"Model class is not callable: {model_cls!r}")

        return model_cls

    @property
    def optimizer_cls(self) -> Callable:
        """Get the optimizer class."""
        optimizer_cls = self.import_obj(self.optimizer_class)
        if not callable(optimizer_cls):
            raise ValueError(f"Optimizer class is not callable: {optimizer_cls!r}")

        return optimizer_cls

    @property
    def splitter_cls(self) -> Callable:
        """Get the splitter class."""
        splitter_cls = self.import_obj(self.splitter_class)
        if not callable(splitter_cls):
            raise ValueError(f"Splitter class is not callable: {splitter_cls!r}")

        return splitter_cls

    @classmethod
    def from_file(
        cls, path: pathlib.Path, workdir: pathlib.Path | None = None
    ) -> TrainingConfig:
        """Read the training config from a YAML file."""
        with path.open() as fh:
            kwargs = yaml.safe_load(fh)
        return cls.from_dict(kwargs, workdir)

    @classmethod
    def from_dict(
        cls, data: dict, workdir: pathlib.Path | None = None
    ) -> TrainingConfig:
        """Construct a training config from a dictionary."""
        config = cls(**data)
        if workdir is not None:
            config.resolve_paths(workdir)
        return config

    @classmethod
    def from_separate_configs(
        cls,
        conf_model: dict,
        conf_splitter: dict,
        workdir: pathlib.Path | None = None,
    ) -> TrainingConfig:
        """Construct a training config from separate configs."""
        data = {
            **conf_model,
            **conf_splitter,
            "seed": 113587,
            "train_all_samples": True,
            "frozen_backbone": False,
            "checkpoint_path_pretrained": None,
            "dataset_name": "<obsolete>",  # TODO: remove this
        }
        data.pop("id")
        return cls.from_dict(data, workdir)

    def resolve_paths(self, workdir: pathlib.Path) -> None:
        """Resolve relative internal paths."""
        # TODO: might need to check if they paths are relative.
        if self.input_csv:
            self.input_csv = workdir / self.input_csv
        if self.checkpoint_path_pretrained is not None:
            self.checkpoint_path_pretrained = workdir / self.checkpoint_path_pretrained
