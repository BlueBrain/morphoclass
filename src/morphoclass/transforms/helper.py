"""Various helper and utility functions used in the data transforms."""
from __future__ import annotations

import functools


def raise_no_attribute(attribute):
    """Raise an error for a missing attribute.

    Parameters
    ----------
    attribute : str
        The missing attribute

    Raises
    ------
    ValueError
        Always raised as intended.
    """
    raise ValueError(f"Data object does not have the {attribute} attribute")


def require_field(field_name):
    """Decorate a function with the check if `data` parameter has given field.

    This decorator can only be used for functions with arguments
    (self, data), and the data object must be instance of the `Data` class.

    Parameters
    ----------
    field_name
        The name of the field to be required in `data`.

    Returns
    -------
    callable
        Decorated function.
    """

    def check_field_decorator(call):
        @functools.wraps(call)
        def call_with_check(self, data):
            if not hasattr(data, field_name):
                raise_no_attribute(field_name)
            return call(self, data)

        return call_with_check

    return check_field_decorator


def attribute_bytes_tree_hash(tree):
    """Compute a heuristic hash for a neurite based on its attributes.

    Parameters
    ----------
    tree : tmd.Tree.Tree
        An apical tree.

    Returns
    -------
    int
        The hash of the given tree.
    """
    attrs = ["x", "y", "z", "d", "t", "p"]
    data = (getattr(tree, attr).tobytes() for attr in attrs)

    return hash(data)
