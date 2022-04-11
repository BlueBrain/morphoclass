"""Branching only using the neurites."""
from __future__ import annotations

from morphoclass.transforms.helper import require_field


class BranchingOnlyNeurites:
    """Extract simplified structure from TMD neurites.

    All neurites in the original data or reduced to branching points only
    """

    @require_field("tmd_neurites")
    def __call__(self, data):
        """Callable for TMD Branching only neurites.

        Parameters
        ----------
        data : torch_geometric.data.data.Data
            Data instance.

        Returns
        -------
        data : torch_geometric.data.data.Data
            Processed data instance.
        """
        data.tmd_neurites = [
            neurite.extract_simplified() for neurite in data.tmd_neurites
        ]

        return data

    def __repr__(self):
        """Representation of the BranchingOnlyNeurites class."""
        return f"{self.__class__.__name__}()"
