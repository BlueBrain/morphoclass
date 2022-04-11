"""Transforms that generate morphology data augmentations.

This implies that the transforms in this module will modify the actual
morphology. Normally it is desired to restore the original morphology after the
forward pass. To achieve this insert the "copy" transform into the transform
pipeline.
"""
from __future__ import annotations

from morphoclass.transforms.augmentors.add_nodes_at_intervals import AddNodesAtIntervals
from morphoclass.transforms.augmentors.add_random_points_to_reduction_mask import (
    AddRandomPointsToReductionMask,
)
from morphoclass.transforms.augmentors.add_section_middle_points import (
    AddSectionMiddlePoints,
)
from morphoclass.transforms.augmentors.apply_node_reduction_masks import (
    ApplyNodeReductionMasks,
)
from morphoclass.transforms.augmentors.branching_only_neurites import (
    BranchingOnlyNeurites,
)
from morphoclass.transforms.augmentors.branching_only_neuron import BranchingOnlyNeuron
from morphoclass.transforms.augmentors.equalize_node_counts import EqualizeNodeCounts
from morphoclass.transforms.augmentors.extract_branching_node_reduction_masks import (
    ExtractBranchingNodeReductionMasks,
)
from morphoclass.transforms.augmentors.orient_apicals import OrientApicals
from morphoclass.transforms.augmentors.orient_neuron import OrientNeuron
from morphoclass.transforms.augmentors.random_jitter import RandomJitter
from morphoclass.transforms.augmentors.random_rotation import RandomRotation
from morphoclass.transforms.augmentors.random_stretching import RandomStretching

__all__ = [
    "BranchingOnlyNeurites",
    "BranchingOnlyNeuron",
    "OrientApicals",
    "OrientNeuron",
    "RandomJitter",
    "RandomRotation",
    "RandomStretching",
    "AddSectionMiddlePoints",
    "ApplyNodeReductionMasks",
    "ExtractBranchingNodeReductionMasks",
    "AddRandomPointsToReductionMask",
    "AddNodesAtIntervals",
    "EqualizeNodeCounts",
]
