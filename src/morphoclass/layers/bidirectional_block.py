"""Implementation of the BidirectionalBlock layer."""
from __future__ import annotations

from torch.nn import Module
from torch_geometric.nn import ChebConv

from morphoclass.layers.cat import Cat


class BidirectionalBlock(Module):
    """A parallel pair of ChebConv layers with opposite edge directions.

    Parameters
    ----------
    c_in : int
        The number of input channels.
    c_out : int
        The number of output channels.
    K : int
        The maximal order of the Chebyshev polynomials
    normalization : {"sym", "rw", None}
        The type of the graph Laplacian normalization. See `ChebConv` for more
        details.
    lambda_max : float
        The value for replacing the highest graph Laplacian eigenvalue with.
        Eigenvalues of the graph Laplacian can only be reliably used for
        undirected graphs. This layer, however, makes most sense for directed
        graphs. In oder to still be able to normalize the graph Laplacian it
        was decided to use a fix value. Experiments show that this approach is
        effective.
    flow : {"target_to_source", "source_to_target"}
        The direction of the message passing flow in the ChebConv layers. This
        is not exactly the same as transposing the adjacency matrix, probably
        due to the asymmetric way the normalization of the graph Laplacian
        works.
    """

    def __init__(
        self,
        c_in,
        c_out,
        K=5,
        normalization="sym",
        lambda_max=3.0,
        flow="target_to_source",
    ):
        super().__init__()
        self.K = K
        self.normalization = normalization
        self.lambda_max = lambda_max
        self.flow = flow

        if c_out % 2 != 0:
            raise ValueError("The output channel size must be even")

        self.conv_1 = ChebConv(
            c_in,
            c_out // 2,
            K=self.K,
            flow="target_to_source",
            normalization=self.normalization,
        )
        self.conv_2 = ChebConv(
            c_in,
            c_out // 2,
            K=self.K,
            flow="target_to_source",
            normalization=self.normalization,
        )
        self.cat = Cat()

    def forward(self, x, edge_index, edge_weight=None):
        """Compute the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            A batch of node features.
        edge_index : torch.Tensor
            A batch of adjacency matrices.
        edge_weight : torch.Tensor
            A batch of edge weights.

        Returns
        -------
        x : torch.Tensor
            The output node features.
        """
        x1 = self.conv_1.forward(
            x, edge_index, edge_weight=edge_weight, lambda_max=self.lambda_max
        )
        x2 = self.conv_2.forward(
            x, edge_index[[1, 0]], edge_weight=edge_weight, lambda_max=self.lambda_max
        )
        x = self.cat([x1, x2])

        return x
