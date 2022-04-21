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
"""Layer for normalization by the standard deviation."""
from __future__ import annotations

import torch
from torch import nn


class RunningStd(nn.Module):
    """Layer for normalization by the standard deviation.

    This layer keeps track of the (biased) standard deviation
    of streaming data at training time. The data is normalized
    using this standard deviation both at training and at
    inference time.

    Parameters
    ----------
    shape : int
        The shape of the input tensor.
    eps : float, optional
        A value added to the standard deviation for numerical
        stability.
    """

    def __init__(self, *shape, eps=1e-05):
        super().__init__()

        self.eps = eps
        self.n = 0
        self.register_buffer("mu", torch.zeros(shape))
        self.register_buffer("d", torch.zeros(shape))
        self.register_buffer("sigma", torch.ones(shape))

    def __call__(self, x):
        """Process a batch of data.

        Parameters
        ----------
        x : torch.Tensor
            A batch of data. Should be of shape (n_batch, *shape)
            where `shape` is the parameter from the constructor.

        Returns
        -------
        x : torch.Tensor
            The normalized batch of data.
        """
        new_n = self.n + x.size()[0]

        if new_n > 0 and self.training:
            with torch.no_grad():
                self.mu: torch.Tensor = (self.n * self.mu + torch.sum(x, dim=0)) / new_n
                self.d: torch.Tensor = (
                    self.n * self.d + torch.sum(x * x, dim=0)
                ) / new_n
                self.n = new_n
                variance = torch.clamp(self.d - self.mu * self.mu, min=0)
                self.sigma = torch.sqrt(variance) + self.eps

        return x / self.sigma
