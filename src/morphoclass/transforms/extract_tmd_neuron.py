"""Implementation of the `ExtractTMDNeuron` transform."""
from __future__ import annotations

from morphoclass.transforms.helper import require_field
from morphoclass.utils import from_morphio_to_tmd


class ExtractTMDNeuron:
    """Convert the MorphIO morphology to the TMD Neuron class."""

    @require_field("morphology")
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
        data.tmd_neuron = from_morphio_to_tmd(data.morphology, remove_duplicates=True)

        return data

    def __repr__(self):
        """Compute the repr."""
        return f"{self.__class__.__name__}()"
