"""Implementation of the edge index (adjacency matrix) extractor."""
from __future__ import annotations

import torch

from morphoclass.transforms.helper import require_field


class ExtractEdgeIndex:
    """Extract the adjacency matrix from apical trees.

    The data should contain the field `tmd_neuron`, see the
    `ExtractTMDNeuron` class.

    For each apical tree in the neuron the the adjacency matrix is extracted
    and saved as `edge_index` field in the `Data` objects. The `edge_index`
    is a sparse representation of the adjacency matrix.

    Parameters
    ----------
    make_undirected : bool
        Symmetrise the adjacency matrix
    """

    def __init__(self, make_undirected=False):
        self.make_undirected = make_undirected

    @require_field("tmd_neurites")
    def __call__(self, data):
        """Apply the transform to given data sample.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The data sample to transform.

        Returns
        -------
        data : torch_geometric.data.Data
            The transformed data sample.
        """
        # Collect all apicals into one graph
        data.edge_index = torch.tensor([[], []], dtype=torch.int64)
        for neurite in data.tmd_neurites:
            adj = neurite.dA.tocoo()
            d_idx = data.edge_index.size()[-1]
            # Directed graph with direction away from soma. This is because
            # in adjacency matrix: rows = children, cols = parents
            sources = torch.tensor(adj.col, dtype=torch.int64) + d_idx
            targets = torch.tensor(adj.row, dtype=torch.int64) + d_idx

            # Make undirected
            if self.make_undirected:
                sources, targets = (
                    torch.cat([sources, targets]),
                    torch.cat([targets, sources]),
                )

            edge_index = torch.stack([sources, targets])
            data.edge_index = torch.cat([data.edge_index, edge_index], dim=1)

        return data

    def __repr__(self):
        """Get representation of class instance."""
        return f"{self.__class__.__name__}(make_undirected={self.make_undirected})"
