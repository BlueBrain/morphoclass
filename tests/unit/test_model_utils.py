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
from __future__ import annotations

import torch

import morphoclass.model_utils as model_utils


def test_hierarchical_labels(capfd):
    label_dict = {
        0: ["TPC", "A"],
        1: ["TPC", "B"],
    }

    # Instantiate from label_dict
    hl = model_utils.HierarchicalLabels.from_flat_labels(label_dict)

    # Roots and labels
    labels = hl.labels
    adj = hl.adj
    roots = hl.roots
    roots = hl.roots  # cached roots
    assert len(roots) == 3
    assert len(labels) == 3
    assert labels[roots] == ["TPC"]
    # print(roots, labels)

    # Test setters
    hl.labels = labels
    out, err = capfd.readouterr()
    assert err.startswith("ERROR")

    hl.adj = adj
    out, err = capfd.readouterr()
    assert err.startswith("ERROR")

    # Class masks
    masks = list(hl.gen_class_masks())
    assert list(masks[0]) == [True, False, False]
    assert list(masks[1]) == [False, True, True]

    # Total class segmentation mask
    class_mask = hl.get_class_segmentation_mask()
    assert list(class_mask) == [0, 1, 1]

    # Flat to hierarchical conversion
    assert list(hl.flat_to_hierarchical_oh(0)) == [1, 1, 0]
    assert list(hl.flat_to_hierarchical_oh(1)) == [1, 0, 1]

    # repr
    assert repr(hl) == f"{hl.__class__.__name__}(roots=['TPC'])"

    # len
    assert len(hl) == 3


def test_hierarchical_labels_deprecated():
    # Data
    t1 = torch.tensor([-1, 0, 0, 0, -1])
    t2 = torch.tensor([-1, 2, 1, 0, -1])
    t3 = torch.tensor([-1, 5, 0, 0, 0])

    # Test `is_tree`
    is_tree = model_utils.HierarchicalLabelsDeprecated.is_tree

    assert is_tree(t1) is True
    assert is_tree(t2) is False
    assert is_tree(t3) is False

    # Test from_labels
    label_dict = {0: ["A", "a"], 1: ["A", "b"], 2: ["A", "c"], 3: ["B"]}
    tree_idx_expect = torch.tensor([-1, 0, 0, 0, -1])
    labels_expect = ["A", "a", "b", "c", "B"]
    one_hot_labels_expect = {
        0: torch.tensor([1, 1, 0, 0, 0]).byte(),
        1: torch.tensor([1, 0, 1, 0, 0]).byte(),
        2: torch.tensor([1, 0, 0, 1, 0]).byte(),
        3: torch.tensor([0, 0, 0, 0, 1]).byte(),
    }

    hl = model_utils.HierarchicalLabelsDeprecated(label_dict=label_dict)
    assert torch.eq(hl.tree_idx, tree_idx_expect).all()
    assert all(l1 == l2 for l1, l2 in zip(hl.labels, labels_expect))
    assert torch.eq(
        hl.get_mask(), torch.tensor([1, 0, 0, 0, 1], dtype=torch.uint8)
    ).all()
    assert torch.eq(
        hl.get_mask(1), torch.tensor([0, 1, 1, 1, 0], dtype=torch.uint8)
    ).all()
    assert torch.eq(
        hl.get_mask(2), torch.tensor([0, 0, 0, 0, 0], dtype=torch.uint8)
    ).all()
    for dense_label, oh_label in one_hot_labels_expect.items():
        assert torch.eq(hl.to_one_hot(dense_label), oh_label).all()
