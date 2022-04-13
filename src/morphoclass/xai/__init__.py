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
"""XAI tools for morphology GNNs."""
from __future__ import annotations

from morphoclass.xai.embedding_extractor import EmbeddingExtractor
from morphoclass.xai.embedding_visualization import get_outlier_detection_app
from morphoclass.xai.grad_cam_explainer import GradCAMExplainer
from morphoclass.xai.grad_cam_on_models import grad_cam_cnn_model
from morphoclass.xai.grad_cam_on_models import grad_cam_gnn_model
from morphoclass.xai.grad_cam_on_models import grad_cam_perslay_model
from morphoclass.xai.model_attributions import cnn_model_attributions
from morphoclass.xai.model_attributions import cnn_model_attributions_population
from morphoclass.xai.model_attributions import gnn_model_attributions
from morphoclass.xai.model_attributions import perslay_model_attributions
from morphoclass.xai.model_attributions import sklearn_model_attributions_shap
from morphoclass.xai.model_attributions import sklearn_model_attributions_tree
from morphoclass.xai.plot_node_saliency import plot_node_saliency

__all__ = [
    "EmbeddingExtractor",
    "GradCAMExplainer",
    "plot_node_saliency",
    "get_outlier_detection_app",
    "get_outlier_detection_app",
    "grad_cam_cnn_model",
    "grad_cam_gnn_model",
    "grad_cam_perslay_model",
    "cnn_model_attributions",
    "gnn_model_attributions",
    "perslay_model_attributions",
    "sklearn_model_attributions_shap",
    "sklearn_model_attributions_tree",
    "cnn_model_attributions_population",
]
