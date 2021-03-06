# Copyright © 2022-2022 Blue Brain Project/EPFL
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
"""Implementation of the hierarchical, bidirectional net (HBNet)."""
from __future__ import annotations

import torch.nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add
from torch_scatter import scatter_max

from morphoclass import layers


class HBNet(torch.nn.Module):
    """Hierarchical, Bidirectional Net.

    Parameters
    ----------
    num_features
        the number of input features
    class_segmentation_mask
        mask for hierarchical labels where each softmax block
        is marked by a different integer. The class mask can be
        generated by the method `get_total_class_mask()` of instances
        of the model_utils.HierarchicalLabels class.
    """

    def __init__(self, num_features, class_segmentation_mask):
        super().__init__()
        self.num_features = num_features
        self.class_mask = torch.from_numpy(class_segmentation_mask)
        self.num_outputs = len(self.class_mask)

        # Note that K=4 in GCN corresponds to K=5 in PyG
        self.conv11 = layers.ChebConv(self.num_features, 64, K=5)
        self.conv12 = layers.ChebConv(self.num_features, 64, K=5)

        self.conv21 = layers.ChebConv(128, 256, K=5)
        self.conv22 = layers.ChebConv(128, 256, K=5)

        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(512, self.num_outputs)

    def _apply(self, fn):
        """Override `Module._apply`.

        This is to make sure that `model.to(device)` also places
        `model.class_mask` on that device.
        """
        super()._apply(fn)
        self.class_mask = fn(self.class_mask)
        return self

    def forward(self, data):
        """Compute the forward pass.

        Parameters
        ----------
        data
            The input data.

        Returns
        -------
        log_softmax
            The log softmax of the predictions.
        """
        # Extract data
        x, edge_index = data.x, data.edge_index
        edge_index_rev = edge_index[[1, 0]]

        # Forward pass
        x1 = F.relu(self.conv11(x, edge_index))
        x2 = F.relu(self.conv12(x, edge_index_rev))
        x = torch.cat([x1, x2], dim=1)

        x1 = F.relu(self.conv21(x, edge_index))
        x2 = F.relu(self.conv22(x, edge_index_rev))
        x = torch.cat([x1, x2], dim=1)

        x = self.pool(x, data.batch)
        x = self.fc(x)

        # Compute the log of the softmax in a hierarchical way
        norm_sparse = torch.log(scatter_add(src=torch.exp(x), index=self.class_mask))
        norm = norm_sparse[:, self.class_mask]
        log_softmax = x - norm

        return log_softmax

    def loss(self, data):
        """Compute the loss.

        Parameters
        ----------
        data
            The input data.

        Returns
        -------
        The loss.
        """
        log_softmax = self.forward(data)
        targets = data.y_oh.type(log_softmax.dtype)
        return torch.mean(-targets * log_softmax)

    def predict_probabilities(self, data):
        """Compute the prediction probabilities.

        Parameters
        ----------
        data
            The input data.

        Returns
        -------
        The prediction probabilities.
        """
        log_softmax = self.forward(data)
        return torch.exp(log_softmax)

    def predict(self, data):
        """Make hierarchical predictions for given data.

        Parameters
        ----------
        data
            A batch of samples.

        Returns
        -------
        val_max
            Probabilities for the predicted nodes in the hierarchy tree.
        val_argmax
            Indices for the predicted nodes in the hierarchy tree.
            For example, hierarchical one-hot prediction `[1, 0, 0, 1, 0]`
            would correspond to `val_argmax = [0, 3]`.

        """
        probabilities = self.predict_probabilities(data)
        val_max, val_argmax = scatter_max(src=probabilities, index=self.class_mask)
        return val_max, val_argmax

    def gen_hierarchical_probabilities(self, probabilities, parent_mask, hl):
        """Descend the hierarchy layer-wise and compute probabilities.

        Starting with the parent nodes specified in `parent_mask` compute
        the probabilities for their children using

            P(child) = P(child|parent) * P(parent),

        create a new mask containing all children, and recurse.

        Parameters
        ----------
        probabilities
            Probabilities for all nodes in the hierarchy tree.
        parent_mask
            Starting mask for the nodes from which to start descending
            the hierarchy. First call of the function should have
            `parent_mask` contain all root nodes.
        hl : model_utils.HierarchicalLabels
            Hierarchy structure for labels.

        Yields
        ------
        mask
            Mask for all nodes considered for the current layer in the
            hierarchy.
        probabilities
            Probabilities predicted for the nodes specified by by mask.
        """
        if len(probabilities.shape) != 2:
            msg = "probabilities must have shape (n_samples, n_labels)"
            raise ValueError(msg)
        else:
            assert probabilities.shape[1] == len(hl)

        # First yield the probabilities for the given mask
        yield parent_mask, probabilities * parent_mask.type_as(probabilities)

        # Calculate the mask and probabilities for the next level.
        new_mask = torch.zeros_like(parent_mask)
        new_probabilities = probabilities.clone()

        # Add all children of parents in parent_mask and recurse
        for parent_idx in parent_mask.nonzero(as_tuple=False).flatten():
            children_mask = hl.adj[:, parent_idx] == 1
            children_mask = torch.tensor(children_mask, dtype=torch.bool)
            if children_mask.sum() == 0:
                # leaf node, keep it
                new_mask[parent_idx] = True
            else:
                # compute P(child) = P(child|parent) * P(parent)
                new_probabilities[:, children_mask] *= probabilities[:, [parent_idx]]
                new_mask |= children_mask
        if all(parent_mask == new_mask):
            # No new children were found, we're finished
            return
        else:
            yield from self.gen_hierarchical_probabilities(
                new_probabilities, new_mask, hl
            )

    def compute_hierarchical_metric(self, metric_function, data, hl):
        """Compute metric for each layer in the label hierarchy.

        Parameters
        ----------
        metric_function
            A function computing the required metric. Should have the
            following signature `metric_function(targets, predictions)`.
        data
            A batch of samples.
        hl : model_utils.HierarchicalLabels
            Hierarchy structure.

        Returns
        -------
        results
            List of metric evaluations for each layer in the label hierarchy.
        """
        results = []
        parent_mask = torch.tensor(hl.roots, dtype=torch.bool)
        flat_proba = self.predict_probabilities(data)
        for mask, proba in self.gen_hierarchical_probabilities(
            flat_proba, parent_mask, hl
        ):
            t = data.y_oh[:, mask]
            p = proba[:, mask]

            assert all(t.sum(dim=1) == 1), "Labels should be hierarchically one-hot"

            results.append(metric_function(t, p))

        return results

    def hierarchical_accuracies(self, data, hl):
        """Accuracies for each layer in the label hierarchy.

        Parameters
        ----------
        data
            A batch of samples.
        hl : model_utils.HierarchicalLabels
            Hierarchy structure.

        Returns
        -------
        accuracies
            List of accuracies for each layer in the label hierarchy.
        """

        def accuracy_fn(targets, predictions):
            t = targets.argmax(dim=1)
            p = predictions.argmax(dim=1)
            return torch.sum(t == p).item() / len(t)

        accuracies = self.compute_hierarchical_metric(accuracy_fn, data, hl)

        return accuracies

    def precision_recall_f1(self, data, hl, *, average="micro"):
        """Compute hierarchical precision, recall, F1 score.

        Parameters
        ----------
        data
            Abatch of samples.
        hl : model_utils.HierarchicalLabels
            Hierarchy structure.
        average
            The type of multi-class average to take, accepted values
            are `None`, `"micro"`, and `"macro"`.

        Returns
        -------
        p_h
            The hierarchical precision.
        r_h
            The hierarchical recall.
        f1_h
            The hierarchical F1 score
        """
        if average not in {None, "micro", "macro"}:
            raise ValueError('average has to be one of {None, "micro", "macro"}')

        # Get model prediction probabilities and true labels
        pred_prob = self.predict_probabilities(data)
        y_h = data.y_oh

        # Construct the augmented predictions by including all ancestors
        parent_mask = torch.tensor(hl.roots, dtype=torch.bool)
        pred_h = torch.zeros_like(y_h)
        to_oh = torch.eye(pred_prob.shape[1], dtype=torch.bool)

        last_mask = None
        for mask, prob in self.gen_hierarchical_probabilities(
            pred_prob, parent_mask, hl
        ):
            pred_h[to_oh[prob.argmax(dim=1).detach()]] = 1
            # `mask` and `prob` of the last item correspond to flat predictions
            # which we need for the macro and None averages
            last_mask = mask
        assert last_mask is not None, "No probabilities were generated"

        # Compute the actual scores
        if average == "micro":
            p_h_all = torch.sum(y_h * pred_h).to(torch.float) / pred_h.sum()
            r_h_all = torch.sum(y_h * pred_h).to(torch.float) / y_h.sum()

            p_h_micro = p_h_all.item()
            r_h_micro = r_h_all.item()
            f1_h_micro = 2 * p_h_micro * r_h_micro / (p_h_micro + r_h_micro)

            return p_h_micro, r_h_micro, f1_h_micro

        # Now it's either 'macro' or 'None'
        # p_h: Union[List[float], float]
        # r_h: Union[List[float], float]
        # f1_h: Union[List[float], float]
        p_h = []
        r_h = []
        f1_h = []

        for i in sorted(hl.flat_labels()):
            # for the given class n get the corresponding index
            # in the masked predictions
            i_idx_t = torch.tensor(hl.flat_to_hierarchical_oh(i))[last_mask]
            i_idx = i_idx_t.argmax().item()
            i_predictions = pred_h[:, last_mask].argmax(dim=1) == i_idx
            i_labels = data.y == i
            i_intersect = (i_predictions == 1) & (i_labels == 1)

            score_intersect = y_h[i_intersect] * pred_h[i_intersect]
            p_h_cls_t = (
                torch.sum(score_intersect).to(torch.float) / pred_h[i_predictions].sum()
            )
            r_h_cls_t = torch.sum(score_intersect).to(torch.float) / y_h[i_labels].sum()

            p_h_cls = p_h_cls_t.item()
            r_h_cls = r_h_cls_t.item()

            p_h.append(p_h_cls)
            r_h.append(r_h_cls)
            if p_h_cls + r_h_cls == 0:
                f1_h.append(0.0)
            else:
                f1_h.append(2 * p_h_cls * r_h_cls / (p_h_cls + r_h_cls))

        if average == "macro":
            return sum(p_h) / len(p_h), sum(r_h) / len(r_h), sum(f1_h) / len(f1_h)
        else:  # average == None
            return p_h, r_h, f1_h
