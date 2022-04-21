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
"""Implementation of the `morphoclass extract-features` CLI command."""
from __future__ import annotations

import logging
import os
import pathlib
import re
from typing import Literal

import click

logger = logging.getLogger(__name__)


@click.command(
    name="extract-features",
    help="""
    Extract morphology features.

    The dataset is read from the given CSV file and processed according to the
    set options. The pre-processing consists of extracting the specified
    neurites, orienting them, and reducing to branching nodes only if the
    corresponding flags are set. Then, depending on the features parameter
    the features are extracted:

    \b
    * graph-rd: graph features with radial distances
    * graph-proj: graph features with distances to the y-axis
      (projection onto the y-axis)
    * diagram-tmd-rd: TMD persistence diagram with radial distances as
      filtration function.
    * diagram-tmd-proj: TMD persistence diagram with y-axis projection features.
    * diagram-deepwalk: persistence diagram with deepwalk features (if deepwalk
      is installed).
    * image-tmd-rd: TMD persistence image with radial distances as
      filtration function.
    * image-tmd-proj: TMD persistence image with y-axis projection features.
    * image-deepwalk: persistence image with deepwalk features (if deepwalk
      is installed).
    """,
)
@click.argument("csv_path", type=click.Path(dir_okay=False))
@click.argument("neurite_type", type=click.Choice(["apical", "axon", "basal", "all"]))
@click.argument(
    "feature",
    type=click.Choice(
        [
            "graph-rd",
            "graph-proj",
            "diagram-tmd-rd",
            "diagram-tmd-proj",
            "diagram-deepwalk",
            "image-tmd-rd",
            "image-tmd-proj",
            "image-deepwalk",
        ]
    ),
)
@click.argument("output_dir", type=click.Path(file_okay=False), required=True)
@click.option(
    "--orient",
    is_flag=True,
    help="Orient the neurons so that the apicals are aligned with the positive y-axis.",
)
@click.option(
    "--no-simplify-graph",
    is_flag=True,
    help="""
    By default the neurite graph is reduced to branching nodes only. With this
    flag the full neurite graph will be preserved.
    """,
)
@click.option(
    "--keep-diagram",
    is_flag=True,
    help="After converting the diagram to persistence image don't discard the diagram.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Don't ask for overwriting existing output files.",
)
def cli(
    csv_path: str | os.PathLike,
    neurite_type: Literal["apical", "axon", "basal", "all"],
    feature: Literal[
        "graph-rd",
        "graph-proj",
        "diagram-tmd-rd",
        "diagram-tmd-proj",
        "diagram-deepwalk",
        "image-tmd-rd",
        "image-tmd-proj",
        "image-deepwalk",
    ],
    output_dir: str | os.PathLike,
    orient: bool,
    no_simplify_graph: bool,
    keep_diagram: bool,
    force: bool,
) -> None:
    """Extract morphology features."""
    output_dir = pathlib.Path(output_dir)
    if output_dir.exists() and not force:
        raise click.ClickException(
            "The output directory already exists. Either delete it manually or "
            "use the --force flag."
        )

    logger.info("Loading modules and libraries")
    import numpy as np
    import torch

    from morphoclass import transforms
    from morphoclass.data.morphology_dataset import MorphologyDataset
    from morphoclass.feature_extractors import non_graph

    logger.info("Starting feature extraction")
    # This transform should always be the last transform of the dataset. It filter out
    # only those data fields that we want to keep for serialisation. In particular,
    # the original morphology and the TMD neurites are discarded.
    data_fields = [
        "edge_index",
        "edge_weight",
        "u",
        "x",
        "y",
        "y_str",
        "label",
        "z",
        "diagram",
        "image",
        "path",
        "__num_nodes__",  # set by data.num_nodes = <value>
    ]
    field_filter = transforms.MakeCopy(data_fields)

    # Orientation, TMD neurites and adjacency matrix
    logger.info("Setting up pre-transforms")
    pre_transforms: list[object] = [
        transforms.ExtractTMDNeurites(neurite_type=neurite_type)
    ]
    if orient:
        pre_transforms.append(transforms.OrientApicals())
    if not no_simplify_graph:  # = if simplify
        pre_transforms.append(transforms.BranchingOnlyNeurites())
    pre_transforms.append(transforms.ExtractEdgeIndex())

    # Load the dataset with the pre-transforms common to all feature types
    logger.info("Loading data")
    dataset = MorphologyDataset.from_csv(
        csv_path,
        pre_transform=transforms.Compose(pre_transforms),
    )

    # Check for morphologies with too little data
    idx_bad = _find_bad_neurites(dataset)
    if idx_bad:
        bad_sample_list = "\n".join(f"* {dataset[idx].path}" for idx in idx_bad)
        logger.error(
            "Some morphologies had neurites with a total neurite node count "
            "less than 3. This is too little for feature extraction and we'll "
            "therefor remove these morphologies from the dataset. Consider "
            "inspecting the data to find the cause. The morphologies to remove "
            f"are:\n{bad_sample_list}"
        )
        idx_good = [idx for idx in range(len(dataset)) if idx not in idx_bad]
        dataset = dataset.index_select(idx_good)

    # Features
    logger.info("Extracting features")
    if feature in {"graph-rd", "graph-proj"}:
        feature_extractor: transforms.ExtractDistances
        if feature == "graph-rd":
            feature_extractor = transforms.ExtractRadialDistances()
        elif feature == "graph-proj":
            feature_extractor = transforms.ExtractVerticalDistances()
        else:
            raise RuntimeError(f"Unknown graph feature: {feature!r}")
        scaler = transforms.FeatureRobustScaler(
            feature_indices=[0], with_centering=False
        )
        dataset.transform = transforms.Compose(
            [transforms.MakeCopy(), feature_extractor]
        )
        scaler.fit(dataset)
        dataset.transform = transforms.Compose(
            [transforms.MakeCopy(), feature_extractor, scaler, field_filter]
        )
    else:  # "diagram-" or "image-"
        # Set num_nodes in data to avoid warnings in data loaders. This is
        # because data is made for graph structures, but we're not using any
        # node features.
        for data in dataset:
            data.num_nodes = sum(len(tree.p) for tree in data.tmd_neurites)

        # Add diagrams
        neurite_collection = [data.tmd_neurites for data in dataset]
        if re.fullmatch(r"(diagram|image)-tmd-rd", feature):
            diagrams = non_graph.get_tmd_diagrams(
                neurite_collection, "radial_distances"
            )
        elif re.fullmatch(r"(diagram|image)-tmd-proj", feature):
            diagrams = non_graph.get_tmd_diagrams(neurite_collection, "projection")
        elif re.fullmatch(r"(diagram|image)-deepwalk", feature):
            diagrams = non_graph.get_deepwalk_diagrams(neurite_collection)
        else:
            raise RuntimeError(f"Unknown feature: {feature!r}")

        # Normalize diagrams
        xmin, ymin = np.stack([d.min(axis=0) for d in diagrams]).min(axis=0)
        xmax, ymax = np.stack([d.max(axis=0) for d in diagrams]).max(axis=0)

        xscale = max(abs(xmax), abs(xmin))
        yscale = max(abs(ymax), abs(ymin))
        scale = np.array([[xscale, yscale]])
        for sample, diagram in zip(dataset, diagrams):
            sample.diagram = torch.tensor(diagram / scale).float()

        if feature.startswith("image-"):
            logger.info("Converting diagrams to images")
            # Find diagrams with fewer than 3 points - these can't be converted
            # to persistence images
            idx_bad = _find_bad_diagrams(diagrams)
            if idx_bad:
                bad_sample_list = "\n".join(f"* {dataset[idx].path}" for idx in idx_bad)
                logger.error(
                    "Some diagrams had fewer than 3 points. This is too few to"
                    "to compute persistence images and we'll therefore "
                    "remove these morphologies from the dataset. Consider "
                    "inspecting the data to find the cause. The morphologies "
                    f"to remove are:\n{bad_sample_list}"
                )
                idx_good = [idx for idx in range(len(dataset)) if idx not in idx_bad]
                dataset = dataset.index_select(idx_good)
                diagrams = [diagrams[idx] for idx in idx_good]

            # Compute persistence images
            from tmd.Topology.analysis import get_persistence_image_data

            # If all values are positive then we'll set xmin = ymin = 0, which
            # seems more natural. In particular this is true if the filtration
            # function is positive definite, e.g. radial distances. Note that
            # this step is not done in the original TMD package.
            xmin_normalized = min(xmin, 0)
            ymin_normalized = min(ymin, 0)

            for sample, diagram in zip(dataset, diagrams):
                image = get_persistence_image_data(
                    diagram,
                    xlims=(xmin_normalized, xmax),
                    ylims=(ymin_normalized, ymax),
                )
                image = np.rot90(image)
                image = image[np.newaxis, np.newaxis]  # shape = (batch, c, w, h)
                # convert to tensor, copy to avoid the negative strides error
                sample.image = torch.tensor(image.copy()).float()

                if not keep_diagram:
                    del sample.diagram
        dataset.transform = field_filter

    # Make the paths relative to the base directory - avoids paths being
    # machine specific
    logger.info("Setting the path attributes")
    base_dir = pathlib.Path.cwd()
    for data in dataset.data:
        data.path = str(pathlib.Path(data.path).resolve().relative_to(base_dir))

    logger.info("Saving extracted features to disk")
    output_dir.mkdir(exist_ok=force, parents=True)
    for sample in dataset:
        file_name = pathlib.Path(sample.path).with_suffix(".features").name
        sample.save(output_dir / file_name)

    logger.info("Done.")


def _find_bad_neurites(dataset):
    """Find neurites with fewer than three nodes."""
    idx_bad = set()
    for idx, sample in enumerate(dataset):
        num_nodes = sum(len(tree.p) for tree in sample.tmd_neurites)
        if num_nodes < 3:
            idx_bad.add(idx)

    return idx_bad


def _find_bad_diagrams(diagrams):
    """Find diagrams with fewer than three points."""
    idx_bad = set()
    for idx, diagram in enumerate(diagrams):
        if len(diagram) < 3:
            idx_bad.add(idx)

    return idx_bad
