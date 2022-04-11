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
