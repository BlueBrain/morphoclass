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
"""Implementation of the tree-LSTM pooling layer."""
from __future__ import annotations

import numpy as np
import torch
from scipy import sparse


class TreeLSTMCell(torch.nn.Module):
    """Child-Sum Tree LSTM Cell.

    This class implements a single child-sum LSTM cell, and can be applied
    to a single node in a tree. (see the implementation of the `forward`
    function).

    WARNING: this implementation is not at all optimized, and ideally one
    should be able to apply LSTM cells to multiple nodes in parallel. The
    current implementation only supports a sequential processing.

    Parameters
    ----------
    x_size : int
        Dimension of the node embedding vector
    h_size : int
        Dimension of the hidden state and the memory cell vectors

    See Also
    --------
    https://arxiv.org/abs/1503.00075
    """

    def __init__(self, x_size, h_size):
        super().__init__()
        self.x_size = x_size
        self.h_size = h_size

        # 3 = (i, o, u)
        w_iou = torch.empty([3, h_size, x_size])
        u_iou = torch.empty([3, h_size, h_size])
        b_iou = torch.empty([3, h_size])

        w_f = torch.empty([h_size, x_size])
        u_f = torch.empty([h_size, h_size])
        b_f = torch.empty([h_size, 1])

        self.w_iou = torch.nn.Parameter(w_iou, requires_grad=True)
        self.u_iou = torch.nn.Parameter(u_iou, requires_grad=True)
        self.b_iou = torch.nn.Parameter(b_iou, requires_grad=True)

        self.w_f = torch.nn.Parameter(w_f, requires_grad=True)
        self.u_f = torch.nn.Parameter(u_f, requires_grad=True)
        self.b_f = torch.nn.Parameter(b_f, requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        """Randomly initialize all weights and set biases to zero."""
        torch.nn.init.xavier_uniform_(self.w_iou)
        torch.nn.init.xavier_uniform_(self.u_iou)
        torch.nn.init.zeros_(self.b_iou)

        torch.nn.init.xavier_uniform_(self.w_f)
        torch.nn.init.xavier_uniform_(self.u_f)
        torch.nn.init.zeros_(self.b_f)

    def forward(self, x, hs, cs):
        """Single forward pass of the LSTM cell on a node.

        Parameters
        ----------
        x : torch.tensor
            The node embedding vector of shape (x_size, )
        hs : torch.tensor
            Hidden states of the child nodes. The shape should be
            (n_children, h_size)
        cs : torch.tensor
            Memory states of the child nodes. The shape should be
            (n_children, h_size)

        Returns
        -------
        h : torch.tensor
            The updated hidden state of shape (h_size, )
        c : torch.tensor
            The updated memory cell of shape (h_size, )
        """
        h_sum = hs.sum(dim=0)

        i, o, u = self.w_iou @ x + self.u_iou @ h_sum + self.b_iou
        fs = (self.w_f @ x).unsqueeze(1) + self.u_f @ hs.T + self.b_f

        i, o, fs = torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(fs)
        u = torch.tanh(u)

        c = i * u + torch.sum(fs * cs.T, dim=1)
        h = o * torch.tanh(c)

        return h, c


class TreeLSTMPool(torch.nn.Module):
    """Child-Sum Tree LSTM Pooling Layer.

    This class implements the Tree-LSTM as described in the reference. After
    traversing the whole tree the hidden state of the root node is returned
    as the result of the pooling.

    Batched graphs can be processed as well. In that case the hidden states
    of all root nodes of all graphs are returned.

    WARNING: this implementation is not at all optimized, and ideally one
    should be able to apply LSTM cells to multiple nodes in parallel. In
    principle all cells yielded by `topologically_sorted` can be processed
    in parallel, but the current implementation sequentially loops through
    them.

    Parameters
    ----------
    x_size : int
        Dimension of the node embedding vector
    h_size : int
        Dimension of the hidden state and the memory cell vectors

    See Also
    --------
    https://arxiv.org/abs/1503.00075
    """

    def __init__(self, x_size, h_size):
        super().__init__()
        self.x_size = x_size
        self.h_size = h_size
        self.cell = TreeLSTMCell(x_size, h_size)

    @staticmethod
    def topologically_sorted(adj):
        """Topologically sort nodes in a given tree.

        Topological sort mean that we start with all leaf nodes and yield
        them. Next all nodes for which all children have already been
        processed are yielded. This is repeated until all nodes have been seen.

        At each iteration step a list of nodes is returned (an equivalent node
        mask to be precise) which are next in the topological order.

        Parameters
        ----------
        adj : matrix_like
            The adjacency matrix describing the tree to be sorted

        Yields
        ------
        active_nodes : array_like of type bool
            Node mask for the current set of nodes in a sorted tree.
        """
        degrees = adj.sum(axis=1).A1
        accumulator = np.zeros_like(degrees)
        active_nodes = (accumulator == degrees).astype(int)
        # set degree of leaf nodes to -1 so that they don't trigger any
        degrees -= active_nodes
        # more
        while any(active_nodes):
            yield active_nodes.astype(bool)
            accumulator += adj @ active_nodes
            active_nodes = (accumulator == degrees).astype(int)
            accumulator -= active_nodes

    def forward(self, x, edge_index):
        """Compute the forward pass.

        Parameters
        ----------
        x
            The batched node feature maps for pooling.
        edge_index
            The batched adjacency matrices.

        Returns
        -------
        h : torch.Tensor
            The pooled features
        """
        device = next(self.parameters()).device
        n_nodes = len(x)
        assert x.shape == (n_nodes, self.x_size)
        h = torch.zeros(n_nodes, self.h_size).to(device)
        c = torch.zeros(n_nodes, self.h_size).to(device)

        adj = sparse.csr_matrix(
            (np.ones(edge_index.shape[1]), edge_index.detach().cpu().numpy()),
            shape=(n_nodes, n_nodes),
            dtype=int,
        )

        for active_nodes in self.topologically_sorted(adj):
            # In principle all active nodes could be handled in parallel,
            # I'm not sure how to implement it though, so for now I'll just
            # cycle through them sequentially...
            for idx in np.where(active_nodes)[0]:
                child_mask = adj[idx].toarray().ravel().astype(bool)
                h[idx], c[idx] = self.cell(x[idx], h[child_mask], h[child_mask])

        roots = np.where(adj.sum(axis=0).A1 == 0)[0]
        return h[roots]


#         return h[active_nodes]
