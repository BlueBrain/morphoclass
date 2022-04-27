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
"""Utilities for working with the NeuroTS package."""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import neurots.extract_input


def tns_distributions_from_dataset(
    dataset,
    ids=None,
    feature="path_distances",
    neurite_types=("basal", "apical", "axon"),
):
    """Extract TNS distributions from morphology dataset.

    It is required that all samples in the dataset have the
    attribute "path", which is used to extract the layer name

    Parameters
    ----------
    dataset : morphoclass.data.morphology_dataset.MorphologyDataset
        morphology dataset for which to compute the TNS distributions
    ids : list, optional
        indices specifying a subset of the given dataset (e.g. the indices of
        the training set) if None then the whole dataset will be used.
    feature : str
        feature for creation of TMD barcodes, see documentation of
        `tns.extract_input.distributions`
    neurite_types : list_like
        neurite type for which to compute the distributions. Possible entries
        are 'apical', 'basal', 'axon'.

    Returns
    -------
    distributions : dict
        a dictionary containing extracted distribution parameters per m-type
    """
    if ids is None:
        ids = range(len(dataset))
    distributions = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx in ids:
            file = pathlib.Path(dataset[idx].path)
            filename = file.name
            layer_str = file.parent.name
            out_file = pathlib.Path(temp_dir) / layer_str / filename
            out_file.parent.mkdir(exist_ok=True)
            shutil.copy(str(file), str(out_file))
        for layer in sorted(pathlib.Path(temp_dir).iterdir()):
            print(f"Generating distributions for {layer.name}...")
            distributions[layer.name] = neurots.extract_input.distributions(
                str(layer), feature=feature, neurite_types=neurite_types
            )

    return distributions


def read_tns_parameters(parameters_file_path):
    """Read TNS parameters from file.

    Parameters
    ----------
    parameters_file_path : string or Path object
        path of the parameters file

    Returns
    -------
    parameters : dict
        a dictionary containing TNS parameters per m-type
    """
    with open(parameters_file_path) as f:
        parameters = json.load(f)
    parameters = {k.replace(":", "_"): v for k, v in parameters.items()}

    return parameters
