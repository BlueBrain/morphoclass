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
"""Implementation of the `GradCAMExplainer` class."""
from __future__ import annotations

import numpy as np
import torch


class GradCAMExplainer:
    """Wrap a GNN model and extract GradCAM data upon forward pass.

    Parameters
    ----------
    model : torch.nn.Module
        A GNN model.
    cam_layer : torch.nn.Module
        A reference to a layer in `model` from which the GradCAM data will
        be extracted.
    """

    def __init__(self, model, cam_layer):
        self.model = model
        self.cam_layer = cam_layer

        self._register_hooks()

        self.filter_out = None
        self.filter_out_grad = None

        self.grad_hook = None
        self.out_hook = None

    def _register_hooks(self):
        def grad_hook(grad):
            self.filter_out_grad = grad.cpu().numpy()

        def out_hook(module, inp, outp):
            if outp.requires_grad:
                self.grad_hook = outp.register_hook(grad_hook)
                self.filter_out = outp.detach().cpu().numpy()

        self.out_hook = self.cam_layer.register_forward_hook(out_hook)

    def get_cam(
        self, sample, loader_cls, cls_idx=None, relu_weights=True, relu_cam=True
    ):
        """Run the forward pass on `sample` and get the GradCAM data.

        Parameters
        ----------
        sample
            A morphology data sample.
        loader_cls : type[morphoclass.data.MorphologyDataLoader],
                     type[morphoclass.data.MorphologyEmbeddingDataLoader]
            A data loader class.
        cls_idx : int (optional)
            The numerical class of the sample. If not provided then the model
            prediction for the class will be used.
        relu_weights : bool (optional)
            If true then a ReLU non-linearity will be applied to GradCAM
            weights. This effectively discards gradients with negative weights.
        relu_cam : bool (optional)
            If true then a ReLU non-linearity will be applied to the GradCAM
            signal. This effectively sets negative GradCAM data to zero.

        Returns
        -------
        logits
            The logits obtained ofter the forward pass on the given sample.
        cam
            The GradCAM data.

        Raises
        ------
        ValueError
            When collected outputs still contain None values, and registered hook
            wasn't able to collect data on model call.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create loader for sample
        loader = loader_cls([sample])
        batch = next(iter(loader)).to(device)

        # Forward prop
        self.model.eval()
        out = self.model.to(device)(batch.to(device))
        logits = out.detach().cpu().numpy().squeeze()
        if cls_idx is None:
            cls_idx = logits.argmax()

        # Backward prop
        one_hot_output = torch.zeros_like(out)
        one_hot_output[0][cls_idx] = 1
        out.backward(gradient=one_hot_output)

        # Comput CAM
        assert self.filter_out is not None
        assert self.filter_out_grad is not None

        weights = self.filter_out_grad.mean(axis=0)  # mean over nodes
        if relu_weights:
            weights = np.maximum(weights, 0)
        cam = np.tensordot(weights, self.filter_out, axes=(0, 1))
        if relu_cam:
            cam = np.maximum(cam, 0)  # relu

        return logits, cam
