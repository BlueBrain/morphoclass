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
"""Dataset of synthetic morphologies generated by TNS."""
from __future__ import annotations

import logging
import os
import pathlib
import shutil
import tempfile

import neurots
import numpy as np
from neurots import extract_input
from torch_geometric.data import Data
from torch_geometric.data import Dataset

logger = logging.getLogger(__name__)


def generate_tns_distributions(dataset, ids=None):
    """Generate morphological distributions given a dataset.

    Parameters
    ----------
    dataset
        Model dataset from which to generate the distributions.
    ids
        Indices specifying a subset of the dataset to be used (e.g.
        just the training set indices). If None, then the whole
        dataset is used.

    Returns
    -------
    TNS distributions.
    """
    # TODO: support generation from Data.morphology
    #       will need to use Data.y or Data.y_str to determine the class
    #       and Data.morphology.write(filename) to write the file.

    if ids is None:
        logger.info("No indices provided, using whole dataset")
        ids = range(len(dataset))
    distributions = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx in ids:
            file = dataset[idx].path
            if file is None:
                logging.warning("Attribute 'file' in data is empty, skipping")
                continue
            filename = file.name
            layer = file.parent.name
            out_file = pathlib.Path(temp_dir) / layer / filename
            out_file.parent.mkdir(exist_ok=True)
            shutil.copy(str(file), str(out_file))
        for layer in sorted(pathlib.Path(temp_dir).iterdir()):
            logger.info(f"Generating distributions for {layer.name}...")
            distributions[layer.name] = extract_input.distributions(str(layer))

    return distributions


class TNSDataset(Dataset):
    """Dataset of synthetic morphologies generated by TNS.

    Parameters
    ----------
    input_distributions
        TNS distributions passed as `input_distributions` to `tns.NeuronGrower`.
    input_parameters
        TNS parameters passed as `input_parameters` to `tns.NeuronGrower`.
    n_samples
        The number of morphologies to synthetize.
    m_types
        The m-types to generate.
    layer
        Restrict to m-types from a given layer.
    verbose
        Print additional information to stdout.
    random_state
        The random seed for reproducibility.
    transform
        The transform to pass to the `Dataset` superclass.
    pre_transform
        The pre-transform to pass to the `Dataset` superclass.
    pre_filter
        The pre-filter to pass to the `Dataset` superclass.
    """

    def __init__(
        self,
        input_distributions,
        input_parameters,
        n_samples,
        m_types=None,
        layer=None,
        verbose=False,
        random_state=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.data = []
        self.verbose = verbose
        self.random_state = random_state

        # Collect all the m-types we need. If m-types are not provided, then
        # the keys of `distributions` dictate which m-types will be used
        if m_types is None:
            self.m_types = sorted(input_distributions)
        else:
            self.m_types = sorted(m_types)

        if layer is not None:
            self.m_types = [
                m_type for m_type in self.m_types if m_type.startswith(layer)
            ]

        # Did we find any classes at all?
        self.class_dict = None
        if len(self.m_types) == 0:
            raise ValueError(
                f"No data corresponding to layer {layer} found in data_path"
            )
        else:
            self.class_dict = {
                n: m_type for n, m_type in enumerate(sorted(self.m_types))
            }
            self.class_dict_inv = {v: k for k, v in self.class_dict.items()}

        self.distributions = input_distributions
        self.parameters = {m_type: input_parameters[m_type] for m_type in self.m_types}
        if isinstance(n_samples, dict):
            self.n_samples = n_samples
        else:
            self.n_samples = {m_type: n_samples for m_type in self.m_types}

        # Parent's constructor
        super().__init__(
            root="",
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    def dump_morphologies(self, target_dir, ext="swc", exist_ok=False):
        """Write morphologies to disk.

        Parameters
        ----------
        target_dir
            The target directory.
        ext
            The target file name extension.
        exist_ok
            Continue even if the target directory already exists.
        """
        # Check file extension
        if ext not in ["asc", "swc", "h5"]:
            raise ValueError(f"Unsupported file extension: {ext}")

        # Check target directory
        target_dir = pathlib.Path(target_dir)
        if not os.access(pathlib.Path(), os.W_OK):
            raise OSError(
                f"Can't write to the given directory ({target_dir.resolve()})"
            )

        # Prepare template for file name
        num_samples = len(self.data)
        n_digits = np.floor(np.log10(num_samples)).astype("int") + 1

        # Start dumping
        if self.verbose:
            print("Dumping morphologies to disk...")
        for i, sample in enumerate(self.data, 1):
            i_str = str(i).rjust(n_digits, "0")
            target_file = target_dir / sample.y_str / f"sample{i_str}.{ext}"
            target_file.parent.mkdir(parents=True, exist_ok=exist_ok)
            sample.morphology.write(str(target_file))
            if self.verbose and i % 50 == 0:
                print(f"  > {i}/{num_samples}")
        if self.verbose:
            print("Done.")

    def len(self):
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.data)

    def _download(self):
        # Data is already local, don't need to download
        pass

    def _process(self):
        # We won't be saving data to disc, do just do the processing part
        self.process()

    @property
    def raw_file_names(self):
        """Get the raw file names (not used).

        Returns
        -------
        An empty list.
        """
        return []

    @property
    def processed_file_names(self):
        """Get the preprocessed file names (not used).

        Returns
        -------
        An empty list.
        """
        return []

    def download(self):
        """Download the dataset (not used)."""
        pass

    def synthesize_samples(self, m_type):
        """Synthesize new samples of a given m-type.

        Parameters
        ----------
        m_type
            The m-type for morphology synthesis.

        Returns
        -------
        list
            The synthesized morphologies.
        """
        if self.verbose:
            print(f"Synthesising {self.n_samples[m_type]} {m_type} morphologies...")
        new_data = []
        for i in range(1, self.n_samples[m_type] + 1):
            grower = neurots.NeuronGrower(
                input_distributions=self.distributions[m_type],
                input_parameters=self.parameters[m_type],
            )
            morphology = grower.grow()

            if morphology is not None:
                sample = Data(
                    y=self.class_dict_inv[m_type], y_str=m_type, morphology=morphology
                )
                new_data.append(sample)

            if self.verbose and i % 50 == 0:
                print(f"  > {i}/{self.n_samples[m_type]}")

        return new_data

    def process(self):
        """Synthesize and pre-process the morphology data."""
        self.data = []  # Get rid of any existing data

        state = np.random.get_state()
        if self.random_state is not None:
            np.random.seed(self.random_state)
        try:
            for m_type in sorted(self.m_types):
                self.data += self.synthesize_samples(m_type)
        finally:
            np.random.set_state(state)

        if self.pre_filter is not None:
            self.data = [data for data in self.data if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.data = [self.pre_transform(data) for data in self.data]

    def get(self, idx):
        """Get a data sample by index.

        Parameters
        ----------
        idx
            The index of the morphology data sample.

        Returns
        -------
        A neuron morphology.
        """
        return self.data[idx]
