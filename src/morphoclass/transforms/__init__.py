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
"""Transformations of `Data` objects.

Some of these transformations return a copy of the `Data` object, others
modify the `Data` object in-place. The former are well-suited for on-the-fly
data augmentation and should be used in the `transforms` field of the
`MorphologyDataset` class. The latter, which modify the `Data` objects
in-place should only be called once per sample and are therefore well-suited
for data pre-processing. Therefore the should be used in the `pre_transforms`
field of the `MorphologyDataset` class.

Note that multiple transformations can be chained together using the
`pytorch_geometric.transforms.Compose` class.
"""
from __future__ import annotations

from morphoclass.transforms.add_one_hot_labels import AddOneHotLabels
from morphoclass.transforms.augmentors import AddNodesAtIntervals
from morphoclass.transforms.augmentors import AddRandomPointsToReductionMask
from morphoclass.transforms.augmentors import AddSectionMiddlePoints
from morphoclass.transforms.augmentors import ApplyNodeReductionMasks
from morphoclass.transforms.augmentors import BranchingOnlyNeurites
from morphoclass.transforms.augmentors import BranchingOnlyNeuron
from morphoclass.transforms.augmentors import EqualizeNodeCounts
from morphoclass.transforms.augmentors import ExtractBranchingNodeReductionMasks
from morphoclass.transforms.augmentors import OrientApicals
from morphoclass.transforms.augmentors import OrientNeuron
from morphoclass.transforms.augmentors import RandomJitter
from morphoclass.transforms.augmentors import RandomRotation
from morphoclass.transforms.augmentors import RandomStretching
from morphoclass.transforms.compose import Compose
from morphoclass.transforms.edge_features import ExtractDistanceWeights
from morphoclass.transforms.edge_features import ExtractEdgeFeatures
from morphoclass.transforms.edge_features import ExtractEdgeIndex
from morphoclass.transforms.extract_tmd_neurites import ExtractTMDNeurites
from morphoclass.transforms.extract_tmd_neuron import ExtractTMDNeuron
from morphoclass.transforms.global_feature_to_label import GlobalFeatureToLabel
from morphoclass.transforms.global_features import AbstractGlobalFeatureExtractor
from morphoclass.transforms.global_features import AverageBranchOrder
from morphoclass.transforms.global_features import AverageRadius
from morphoclass.transforms.global_features import ExtractMaximalApicalPathLength
from morphoclass.transforms.global_features import ExtractNumberBranchPoints
from morphoclass.transforms.global_features import ExtractNumberLeaves
from morphoclass.transforms.global_features import TotalPathLength
from morphoclass.transforms.make_copy import MakeCopy
from morphoclass.transforms.node_features import ExtractBranchingAngles
from morphoclass.transforms.node_features import ExtractConstFeature
from morphoclass.transforms.node_features import ExtractCoordinates
from morphoclass.transforms.node_features import ExtractDiameters
from morphoclass.transforms.node_features import ExtractDistances
from morphoclass.transforms.node_features import ExtractIsBranching
from morphoclass.transforms.node_features import ExtractIsIntermediate
from morphoclass.transforms.node_features import ExtractIsLeaf
from morphoclass.transforms.node_features import ExtractIsRoot
from morphoclass.transforms.node_features import ExtractOneHotProperty
from morphoclass.transforms.node_features import ExtractPathDistances
from morphoclass.transforms.node_features import ExtractRadialDistances
from morphoclass.transforms.node_features import ExtractVerticalDistances
from morphoclass.transforms.scalers import AbstractFeatureScaler
from morphoclass.transforms.scalers import FeatureManualScaler
from morphoclass.transforms.scalers import FeatureMinMaxScaler
from morphoclass.transforms.scalers import FeatureRobustScaler
from morphoclass.transforms.scalers import FeatureStandardScaler
from morphoclass.transforms.scalers import scaler_from_config
from morphoclass.transforms.zero_out_features import ZeroOutFeatures

__all__ = [
    # Augmentors
    "BranchingOnlyNeuron",
    "BranchingOnlyNeurites",
    "OrientApicals",
    "OrientNeuron",
    "RandomStretching",
    "RandomRotation",
    "RandomJitter",
    "AddSectionMiddlePoints",
    "ApplyNodeReductionMasks",
    "ExtractBranchingNodeReductionMasks",
    "AddRandomPointsToReductionMask",
    "AddNodesAtIntervals",
    "EqualizeNodeCounts",
    # Global Feature Extractors
    "AbstractGlobalFeatureExtractor",
    "ExtractNumberLeaves",
    "ExtractNumberBranchPoints",
    "ExtractMaximalApicalPathLength",
    "TotalPathLength",
    "AverageBranchOrder",
    "AverageRadius",
    # Node Feature Extractors
    "ExtractBranchingAngles",
    "ExtractConstFeature",
    "ExtractCoordinates",
    "ExtractDiameters",
    "ExtractDistances",
    "ExtractIsBranching",
    "ExtractIsIntermediate",
    "ExtractIsLeaf",
    "ExtractIsRoot",
    "ExtractOneHotProperty",
    "ExtractPathDistances",
    "ExtractRadialDistances",
    "ExtractVerticalDistances",
    # Edge Feature Extractors
    "ExtractEdgeIndex",
    "ExtractEdgeFeatures",
    "ExtractDistanceWeights",
    # Scalers
    "AbstractFeatureScaler",
    "FeatureStandardScaler",
    "FeatureMinMaxScaler",
    "FeatureManualScaler",
    "FeatureRobustScaler",
    "scaler_from_config",
    # Other
    "Compose",
    "ExtractTMDNeuron",
    "ExtractTMDNeurites",
    "AddOneHotLabels",
    "MakeCopy",
    "ZeroOutFeatures",
    "GlobalFeatureToLabel",
]
