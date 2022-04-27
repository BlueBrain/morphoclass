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
"""Various filters used for morphology data loading.

A typical use of filter is in constructing instances of the
`morphoclass.data.MorphologyDataset` class. The filters can
be passed under the `pre_filter` argument in the constructor.
"""
from __future__ import annotations

import functools
import logging
import pathlib

import neurom as nm

logger = logging.getLogger(__name__)


def attribute_check(attribute, default_return):
    """Decorate filter function to check for an attribute in data.

    The filter function should have the signature `filter_fn(data)`,
    where `data` is an instance of `torch_geometric.data.Data`.

    The decorated filter function will check if the `data` object has
    the given attribute, and if it doesn't then a warning is issued
    and the `default_return` will be returned.

    Parameters
    ----------
    attribute : str
        The attribute to check for in `data`.
    default_return : bool
        The default value to be returned by the decorated filter
        function in case where the attribute is missing in `data`.

    Returns
    -------
    wrapped_filter : function
    """

    def file_check_decorator(filter_fn):
        @functools.wraps(filter_fn)
        def wrapped_filter(data):
            if hasattr(data, attribute) and data.path is not None:
                return filter_fn(data)
            else:
                logger.warning(
                    "Filter could not be applied - "
                    f"data has no '{attribute}' attribute."
                )
                return default_return

        return wrapped_filter

    return file_check_decorator


@functools.singledispatch
def exclusion_filter(arg):
    """Construct an exclusion filter for morphology dataset loaders.

    This is a single dispatch class, therefore the concrete
    implementation depends on the type of the `arg` argument.

    Parameters
    ----------
    arg : The argument for the concrete implementation of the filter

    Returns
    -------
    filter_implementation : function
        The  filter function with the signature
        `filter_implementation(data)`, which return True if the `data`
        object should be loaded, and False otherwise.
    """
    return lambda: True


def _clean_filenames(filenames):
    """Clean filenames by excluding comments and empty entries.

    Usually the items in `filenames` are read from a file.
    To allow for commenting out lines in such a file all items in
    `filenames` are pre-processed to remove all comments, i.e. all
    substrings starting with a '#' character.

    Additionally all space-like characters at the beginning and end
    of filenames are removed and duplicate filenames are discarded.

    Parameters
    ----------
    filenames : iterable
        A collections of filenames to be cleaned

    Returns
    -------
    cleaned_filenames : set
        A set containing all unique cleaned filenames.
    """
    cleaned_filenames = set()
    for filename in filenames:
        filename_cleaned, *_ = filename.partition("#")
        filename_cleaned = filename_cleaned.strip()
        if len(filename_cleaned) > 0:
            cleaned_filenames.add(filename_cleaned)

    return cleaned_filenames


@exclusion_filter.register(list)
def filename_exclusion_filter(filenames):
    """Exclusion filter based on a list of filenames.

    Parameters
    ----------
    filenames : list
        A list of filenames to exclude.

    Returns
    -------
    filter_implementation : function
        The filter function.
    """
    excluded_samples = _clean_filenames(filenames)

    @attribute_check("path", default_return=True)
    def filter_implementation(data):
        if pathlib.Path(data.path).stem in excluded_samples:
            return False
        return True

    return filter_implementation


@exclusion_filter.register(str)
def mtype_exclusion_filter(mtype_substring):
    """Exclusion filter based on the m-type of sample.

    It is assumed that the morphology files are organised
    in folders, each folder representing an m-type. If
    the `mtype_substring` is part of the m-type folder
    name of the given `data` instance, then this instance
    is ignored.

    Parameters
    ----------
    mtype_substring : str
        All m-types that contain `mtype_substring` are excluded.

    Returns
    -------
    filter_implementation : function
        The implementation of the filter.
    """

    @attribute_check("path", default_return=True)
    def filter_implementation(data):
        if mtype_substring in pathlib.Path(data.path).parent.name.upper():
            return False
        return True

    return filter_implementation


@functools.singledispatch
def inclusion_filter(arg):
    """Construct an inclusion filter for morphology dataset loaders.

    A typical use of filter is in constructing instances of the
    `morphoclass.data.MorphologyDataset` class. The filters can
    be passed under the `pre_filter` argument in the constructor.

    This is a single dispatch class, therefore the concrete
    implementation depends on the type of the `arg` argument.

    Parameters
    ----------
    arg : The argument for the concrete implementation of the filter

    Returns
    -------
    filter_implementation : function
        The  filter function with the signature
        `filter_implementation(data)`, which return True if the `data`
        object should be loaded, and False otherwise.
    """
    return lambda: True


@inclusion_filter.register(list)
def filename_inclusion_filter(filenames):
    """Exclusion filter based on a list of filenames.

    Parameters
    ----------
    filenames : list
        A list of filenames to include.

    Returns
    -------
    filter_implementation : function
        The filter function.
    """
    included_samples = _clean_filenames(filenames)

    @attribute_check("path", default_return=False)
    def filter_implementation(data):
        if pathlib.Path(data.path).stem in included_samples:
            return True
        return False

    return filter_implementation


@attribute_check("morphology", default_return=False)
def has_apicals_filter(data):
    """Keep only data objects with morphologies containing apical dendrites.

    Returns
    -------
    keep : bool
        Whether to keep or not to keep the data object.
    """
    if any(
        neurite.type == nm.NeuriteType.apical_dendrite
        for neurite in data.morphology.neurites
    ):
        return True
    return False


def combined_filter(*filters):
    """Combine multiple data filters into one.

    Parameters
    ----------
    filters : function
        A list of filter functions to be combined.

    Returns
    -------
    filter_fn : function
        A filter function combining all filters in `filters`.

    Examples
    --------
        >>> filter_1 = data_exclusion_filter("IPC")
        >>> filter_2 = data_exclusion_filter(["file1", "file2"])
        >>> my_filter = combined_filter(filter_1, filter_2)
    """

    def filter_implementation(data):
        return all(fn(data) for fn in filters)

    return filter_implementation
