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
"""Various models for morphology type classification of neurons."""
from __future__ import annotations

from morphoclass.models.bidirectional_net import BidirectionalNet
from morphoclass.models.cnnet import CNNEmbedder
from morphoclass.models.cnnet import CNNet
from morphoclass.models.concatecnnet import ConcateCNNet
from morphoclass.models.concatenet import ConcateNet
from morphoclass.models.coriander_net import CorianderNet
from morphoclass.models.hbnet import HBNet
from morphoclass.models.man_net import ManEmbedder
from morphoclass.models.man_net import ManNet
from morphoclass.models.man_net import ManNetR
from morphoclass.models.man_res_nets import ManResNet1
from morphoclass.models.man_res_nets import ManResNet2
from morphoclass.models.man_res_nets import ManResNet3
from morphoclass.models.multi_adj_net import MultiAdjNet

__all__ = [
    "MultiAdjNet",
    "BidirectionalNet",
    "CNNEmbedder",
    "CNNet",
    "HBNet",
    "ManEmbedder",
    "ManNetR",
    "ManNet",
    "ManResNet1",
    "ManResNet2",
    "ManResNet3",
    "CorianderNet",
    "ConcateNet",
    "ConcateCNNet",
]
