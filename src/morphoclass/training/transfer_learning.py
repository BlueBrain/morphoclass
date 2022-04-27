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
"""Implementation of the transfer learning curves command."""
from __future__ import annotations

import gc
import logging
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from morphoclass.data.morphology_data import MorphologyData
from morphoclass.data.morphology_dataset import MorphologyDataset
from morphoclass.training.cli import run_training
from morphoclass.training.training_config import TrainingConfig
from morphoclass.training.training_log import TrainingLog

logger = logging.getLogger(__name__)


def transfer_learning_curves(
    input_csv,
    features_dir,
    image_path,
    checkpoints_directory,
    model_class,
    model_params,
    dataset_name,
    feature_extractor_name,
    optimizer_class,
    optimizer_params,
    n_epochs,
    batch_size,
    seed,
    checkpoint_path_pretrained,
    frozen_backbone=False,
):
    """Generate TL curves by iterating through splits of different sizes."""
    splitter_class = "sklearn.model_selection.StratifiedKFold"
    checkpoints_directory = pathlib.Path(checkpoints_directory)
    checkpoints_directory.mkdir(exist_ok=True, parents=True)
    dataframe_path = checkpoints_directory / checkpoint_path_pretrained.stem
    dataframe_path = dataframe_path.with_suffix(".csv")
    logger.info(f"Dataframe saved in: {dataframe_path}")

    image_path = pathlib.Path(image_path)
    image_path.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving image in: {image_path}")

    n_splits_list = [2, 3, 4, 5, 6]
    metrics = ["f1_micro", "f1_macro", "f1_weighted", "val_acc_final"]
    supported_ext = [".png", ".eps", ".pdf"]

    df = pd.DataFrame(
        columns=[
            "train_set_size",
            "pretrained",
            "metric",
            "1-score",
        ]
    )

    # Reconstruct the dataset from features
    data = []
    # Sorting to ensure reproducibility
    # TODO: consider tracking the order using a CSV like for datasets
    for path in sorted(pathlib.Path(features_dir).glob("*.features")):
        data.append(MorphologyData.load(path))
    dataset = MorphologyDataset(data)

    for n_splits in n_splits_list:
        logger.info(f"✔ n_splits: {n_splits}/{len(dataset)}")
        splitter_params = {"n_splits": n_splits}

        # with pre-training
        training_config = TrainingConfig(
            input_csv=input_csv,
            model_class=model_class,
            model_params=model_params,
            splitter_class=splitter_class,
            splitter_params=splitter_params,
            dataset_name=dataset_name,
            features_dir=feature_extractor_name,  # TODO put the actual directory
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            n_epochs=n_epochs,
            batch_size=batch_size,
            seed=seed,
            oversampling=False,
            train_all_samples=False,
            checkpoint_path_pretrained=checkpoint_path_pretrained,
            frozen_backbone=frozen_backbone,
        )
        results2 = run_training(dataset, training_config)
        results2.save(
            checkpoints_directory / f"n_splits_{n_splits}_with_pretraining.chk"
        )

        for metric in metrics:
            val_score2 = np.array([split[metric] for split in results2.split_history])
            for score in val_score2:
                df = df.append(
                    {
                        "train_set_size": len(results2.split_history[0]["train_idx"]),
                        "pretrained": "with pre-training",
                        "metric": metric,
                        "1-score": 1 - score,
                    },
                    ignore_index=True,
                )

        del results2
        gc.collect()

        # without pre-training
        checkpoint_path1 = (
            checkpoints_directory / f"n_splits_{n_splits}_without_pretraining.chk"
        )
        if checkpoint_path1.is_file():
            logger.info(f"File w/o pretraining exists, just read: {checkpoint_path1}")
            results1 = TrainingLog.load(checkpoint_path1)
        else:
            training_config = TrainingConfig(
                input_csv=input_csv,
                model_class=model_class,
                model_params=model_params,
                splitter_class=splitter_class,
                splitter_params=splitter_params,
                dataset_name=dataset_name,
                features_dir=feature_extractor_name,  # TODO put the actual directory
                optimizer_class=optimizer_class,
                optimizer_params=optimizer_params,
                n_epochs=n_epochs,
                batch_size=batch_size,
                seed=seed,
                oversampling=False,
                train_all_samples=False,
                checkpoint_path_pretrained=None,
                frozen_backbone=False,
            )
            results1 = run_training(dataset, training_config)
            results1.save(checkpoint_path1)
        for metric in metrics:
            val_score1 = np.array([split[metric] for split in results1.split_history])
            for score in val_score1:
                df = df.append(
                    {
                        "train_set_size": len(results1.split_history[0]["train_idx"]),
                        "pretrained": "without pre-training",
                        "metric": metric,
                        "1-score": 1 - score,
                    },
                    ignore_index=True,
                )
        del results1
        gc.collect()

        df.metric = df.metric.apply(lambda x: "accuracy" if x == "val_acc_final" else x)
        df.to_csv(dataframe_path, index=False)

        fig = Figure(figsize=(16, 5), dpi=75)
        ax = fig.subplots()
        g = sns.catplot(
            x="train_set_size",
            y="1-score",
            hue="pretrained",
            col="metric",
            capsize=0.2,
            palette="YlGnBu_d",
            height=6,
            aspect=0.75,
            kind="point",
            ax=ax,
            data=df,
        )
        g.despine(left=True)
        g.set(ylim=(0, 1), yscale="log", xlabel="Training set size", ylabel="1-score")

        for ext in supported_ext:
            g.savefig(image_path.with_suffix(ext))

        del fig
        gc.collect()
