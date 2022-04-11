"""Implementation of the `ManResNet1`, `ManResNet2`, and `ManResNet3` models."""
from __future__ import annotations

from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn.functional import log_softmax
from torch.nn.functional import nll_loss

from morphoclass import layers


class ManResNet1(Module):
    """MultiAdjNet with 1 residual block.

    In comparison to the MultiAdjNet the second convolution layer is
    replaced by a BidirectionalResBlock with the same input and output
    dimension.

    Because a bidirectional residual block contains two convolutional layers
    the net has a total of three convolutional layers with K=5 ChebConvs.
    Because a K=5 ChebConv takes into account up to the 4th power of the
    adjacency matrix this net has a total reach of 3 * 4 = 12 hops.

    Additionally the AttentionGlobalPool is now the default pooling method.

    Parameters
    ----------
    n_features : int (optional)
        The number of input features.
    n_classes : int (optional)
        The number of output classes.
    """

    def __init__(self, n_features=1, n_classes=4):
        super().__init__()
        self.bidirect = layers.BidirectionalBlock(n_features, 128)
        self.bidirect_res = layers.BidirectionalResBlock(128, 512)
        self.relu = ReLU()
        self.pool = layers.AttentionGlobalPool(512)
        self.fc = Linear(512, n_classes)

    def forward(self, data):
        """Run the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        torch.Tensor
            The log of the prediction probabilities. The tensor has shape
            (n_samples, n_classes).
        """
        x, edge_index = data.x, data.edge_index

        x = self.bidirect(x, edge_index)
        x = self.relu(x)
        x = self.bidirect_res(x, edge_index)
        x = self.relu(x)
        x = self.pool(x, data.batch)
        x = self.fc(x)

        return log_softmax(x, dim=1)

    def loss_acc(self, data):
        """Run the forward pass and compute the loss and accuracy.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        loss : float
            The loss on the given data batch.
        acc : float
            The accuracy on the current data batch.
        """
        # Predictions
        out = self(data)

        # Loss
        loss = nll_loss(out, data.y).item()

        # Accuracy
        pred = out.argmax(dim=1)
        correct = float(pred.eq(data.y).sum().item())
        acc = correct / len(data.y)

        return loss, acc


class ManResNet2(Module):
    """MultiAdjNet with 2 residual blocks.

    In comparison to the MultiAdjNet the first convolutional layer is split
    into a BidirectionalBlock and a BidirectionalResBlock, and the second
    convolutional layer is replaced by another Bidirectional block.

    Because a bidirectional residual block contains two convolutional layers
    the net has a total of five convolutional layers with K=5 ChebConvs.
    Because a K=5 ChebConv takes into account up to the 4th power of the
    adjacency matrix this net has a total reach of 5 * 4 = 20 hops.

    Additionally the AttentionGlobalPool is now the default pooling method.

    Parameters
    ----------
    n_features : int (optional)
        The number of input features.
    n_classes : int (optional)
        The number of output classes.
    """

    def __init__(self, n_features=1, n_classes=4):
        super().__init__()
        self.bidirect = layers.BidirectionalBlock(n_features, 32)
        self.bidirect_res1 = layers.BidirectionalResBlock(32, 128)
        self.bidirect_res2 = layers.BidirectionalResBlock(128, 512)
        self.relu = ReLU()
        self.pool = layers.AttentionGlobalPool(512)
        self.fc = Linear(512, n_classes)

    def forward(self, data):
        """Run the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        torch.Tensor
            The log of the prediction probabilities. The tensor has shape
            (n_samples, n_classes).
        """
        x, edge_index = data.x, data.edge_index

        x = self.bidirect(x, edge_index)
        x = self.relu(x)
        x = self.bidirect_res1(x, edge_index)
        x = self.relu(x)
        x = self.bidirect_res2(x, edge_index)
        x = self.relu(x)
        x = self.pool(x, data.batch)
        x = self.fc(x)

        return log_softmax(x, dim=1)

    def loss_acc(self, data):
        """Run the forward pass and compute the loss and accuracy.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        loss : float
            The loss on the given data batch.
        acc : float
            The accuracy on the current data batch.
        """
        # Predictions
        out = self(data)

        # Loss
        loss = nll_loss(out, data.y).item()

        # Accuracy
        pred = out.argmax(dim=1)
        correct = float(pred.eq(data.y).sum().item())
        acc = correct / len(data.y)

        return loss, acc


class ManResNet3(Module):
    """MultiAdjNet with 3 residual blocks.

    In comparison to the MultiAdjNet the first convolutional layer is split
    into a BidirectionalBlock and two BidirectionalResBlocks, and the second
    convolutional layer is replaced by another Bidirectional block.

    Because a bidirectional residual block contains two convolutional layers
    the net has a total of seven convolutional layers with K=5 ChebConvs.
    Because a K=5 ChebConv takes into account up to the 4th power of the
    adjacency matrix this net has a total reach of 7 * 4 = 28 hops.

    Additionally the AttentionGlobalPool is now the default pooling method.

    Parameters
    ----------
    n_features : int (optional)
        The number of input features.
    n_classes : int (optional)
        The number of output classes.
    """

    def __init__(self, n_features=1, n_classes=4):
        super().__init__()
        self.bidirect = layers.BidirectionalBlock(n_features, 8)
        self.bidirect_res1 = layers.BidirectionalResBlock(8, 32)
        self.bidirect_res2 = layers.BidirectionalResBlock(32, 128)
        self.bidirect_res3 = layers.BidirectionalResBlock(128, 512)
        self.relu = ReLU()
        self.pool = layers.AttentionGlobalPool(512)
        self.fc = Linear(512, n_classes)

    def forward(self, data):
        """Run the forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        torch.Tensor
            The log of the prediction probabilities. The tensor has shape
            (n_samples, n_classes).
        """
        x, edge_index = data.x, data.edge_index

        x = self.bidirect(x, edge_index)
        x = self.relu(x)
        x = self.bidirect_res1(x, edge_index)
        x = self.relu(x)
        x = self.bidirect_res2(x, edge_index)
        x = self.relu(x)
        x = self.bidirect_res3(x, edge_index)
        x = self.relu(x)
        x = self.pool(x, data.batch)
        x = self.fc(x)

        return log_softmax(x, dim=1)

    def loss_acc(self, data):
        """Run the forward pass and compute the loss and accuracy.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        loss : float
            The loss on the given data batch.
        acc : float
            The accuracy on the current data batch.
        """
        # Predictions
        out = self(data)

        # Loss
        loss = nll_loss(out, data.y).item()

        # Accuracy
        pred = out.argmax(dim=1)
        correct = float(pred.eq(data.y).sum().item())
        acc = correct / len(data.y)

        return loss, acc
