"""Implementation of the `OrientNeuron` transform."""
from __future__ import annotations

from morphoclass.orientation import fit_tree_ray
from morphoclass.orientation import orient_neuron
from morphoclass.transforms.helper import require_field


class OrientNeuron:
    """Orient neuron using ray fitting."""

    @require_field("tmd_neuron")
    def __call__(self, data):
        """Apply the morphology transformation.

        Parameters
        ----------
        data
            The input morphology data sample.

        Returns
        -------
        data
            The modified morphology data sample.
        """
        data.tmd_neuron = orient_neuron(fit_tree_ray, data.tmd_neuron, in_place=True)

        return data
