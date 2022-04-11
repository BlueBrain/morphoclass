"""Implementation of the `BranchingOnlyNeuron` transform."""
from __future__ import annotations

from morphoclass.transforms.helper import require_field


class BranchingOnlyNeuron:
    """Extract simplified structure from TMD neurons.

    All neurites in the original data or reduced to branching points only
    """

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
        neuron = data.tmd_neuron
        neuron.apical = [tree.extract_simplified() for tree in neuron.apical]
        neuron.axon = [tree.extract_simplified() for tree in neuron.axon]
        neuron.basal = [tree.extract_simplified() for tree in neuron.basal]
        neuron.undefined = [tree.extract_simplified() for tree in neuron.undefined]

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}()"
