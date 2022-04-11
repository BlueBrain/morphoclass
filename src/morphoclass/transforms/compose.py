"""Implementation of the `Compose` transform."""
from __future__ import annotations


class Compose:
    """A composition of multiple transforms.

    Parameters
    ----------
    transforms : iterable
        The transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

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
        for transform in self.transforms:
            data = transform(data)
        return data

    def __repr__(self):
        """Compute the repr."""
        args_str = "\n".join(f"\t{transform}," for transform in self.transforms)
        return f"{self.__class__.__name__}([\n{args_str}\n])"
