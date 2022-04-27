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
"""Morphology embedding data loader."""
from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


class MorphologyEmbeddingDataLoader(DataLoader):
    """A data loader for the embedded morphology data set.

    This class is derived from `torch.utils.data.DataLoader` and unlike
    `torch_geometric.data.DataLoader` is able to handle
    `data.morphology_dataset.MorphologyEmbedding` objects with non-numeric fields.
    These fields are simply ignored upon constructing batches.

    Parameters
    ----------
    dataset
        The data set to apply the data loader to.
    batch_size
        The batch size.
    shuffle
        Whether to shuffle the data or not.
    **kwargs
        Further keyword arguments to pass on to superclass.
    """

    def __init__(self, dataset, batch_size=None, shuffle=False, **kwargs):
        batch_size = batch_size or len(dataset)
        collate_fn = partial(self.collate_lowerdim_data)
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )

    @staticmethod
    def collate_lowerdim_data(data_list):
        """Collate a list of embedded morphology sample to a batch.

        This method is derived from torch_geometric.data.Batch, and
        modified to ignore keys in samples that cannot be collated.

        Parameters
        ----------
        data_list
            List of `data.morphology_dataset.MorphologyEmbedding` objects to be collated
            to a batch.

        Returns
        -------
        batch
            The batch of collated data objects form `data_list`.
        """
        keys: set[str] = set()
        for data in data_list:
            keys = keys.union(data.keys)
        assert "batch" not in keys

        batch = Batch()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}

        # Initialise all batch keys with empty lists
        for key in keys:
            batch[key] = []

        idx_shifts = {key: 0 for key in keys}
        for data in data_list:
            for key in data.keys:  # Iterate keys in sample
                if isinstance(data[key], np.ndarray):
                    data[key] = torch.from_numpy(data[key].copy()).float()
                # Ignore keys which cannot be collated
                if not (
                    torch.is_tensor(data[key])
                    or isinstance(data[key], int)
                    or isinstance(data[key], float)
                ):
                    continue

                item = data[key] + idx_shifts[key]
                if torch.is_tensor(data[key]):
                    size = data[key].size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                idx_shifts[key] += data.__inc__(key, item)
                batch[key].append(item)

        # Cat lists of tensors / ints / floats into tensors
        for key in batch.keys:
            if key == "embedding":
                batch["point_index"] = []
                for i, embedding in enumerate(batch[key]):
                    batch["point_index"] += [i] * len(embedding)
                batch[key] = torch.cat(batch[key], dim=0)
            else:
                # Remove keys for which not data was collected
                if key.startswith("__") or len(batch[key]) == 0:
                    batch[key] = None
                    continue

                # Cat lists to tensors
                item = batch[key][0]
                if torch.is_tensor(item):
                    batch[key] = torch.cat(
                        batch[key], dim=data_list[0].__cat_dim__(key, item)
                    )
                elif isinstance(item, int) or isinstance(item, float):
                    batch[key] = torch.tensor(batch[key])
                else:
                    raise ValueError("Unsupported attribute type")

        if batch["point_index"] is not None:
            batch["point_index"] = torch.tensor(batch["point_index"])

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()
