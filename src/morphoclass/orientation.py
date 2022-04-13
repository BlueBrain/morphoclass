# Copyright © 2022 Blue Brain Project/EPFL
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
"""Tools for orienting neurons in their 3D embedding vector space."""
from __future__ import annotations

import logging
import pathlib
import shutil
import sys

import numpy as np
import scipy.optimize
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def get_angle_rad(v1, v2):
    """Compute the angle in radians between two given vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        The angle in radians between `v1` and `v2`.
    """
    cos_angle = (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    return np.arccos(cos_angle)


def get_angle_deg(v1, v2):
    """Compute the angle in degrees between two given vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        The angle in degrees between `v1` and `v2`.
    """
    return get_angle_rad(v1, v2) * 180 / np.pi


def get_tree_pca(tree, kind="3d", shift_to_origin=False):
    """Compute the PCA components of the given TMD tree.

    Parameters
    ----------
    tree : tmd.Tree.Tree.Tree
        A TMD tree.
    kind : {"2d", "3d"}
        The kind of PCA to compute. The 2D PCA only considers the x and y
        coordinates, while the 3D uses x, y and z.
    shift_to_origin : bool
        If true shift the tree root to coordinate origin.

    Returns
    -------
    mean : np.array
        The mean of tree coordinates.
    components : np.array
        The three PCA components of the tree.
    explained_variance_ratio : np.array
        The explained variance ratio.

    Raises
    ------
    ValueError
        If the parameter `kind` has an invalid value.
        If the tree does not have exactly one root node.
    """
    if kind == "2d":
        n_components = 2
        coords = [tree.x, tree.y]
    elif kind == "3d":
        n_components = 3
        coords = [tree.x, tree.y, tree.z]
    else:
        msg = "The type parameter must be one of the following: " "('2d', '3d')"
        raise ValueError(msg)
    coords = np.transpose(coords)

    root_id = np.nonzero(tree.p == -1)[0]
    if len(root_id) != 1:
        msg = (
            "the tree provided must have exactly one root, "
            f"but {len(root_id)} roots were found"
        )
        raise ValueError(msg)

    if shift_to_origin:
        coords = coords - coords[root_id]

    pca = PCA(n_components=n_components)
    pca.fit(coords)

    return pca.mean_, pca.components_, pca.explained_variance_ratio_


def get_tree_origin(tree):
    """Find the index of the origin node of a tree.

    Parameters
    ----------
    tree : tmd.Tree.Tree.Tree
        A TMD tree.

    Returns
    -------
    origin : np.ndarray
        The coordinates of the tree origin.
    origin_idx : int
        The index of the origin node.
    """
    origin_idx = np.nonzero(tree.p == -1)[0]
    if len(origin_idx) == 0:
        logger.error("tree has no origin, so it's not a tree.")
        return None, None
    elif len(origin_idx) > 1:
        logger.error("tree has more than one origin, so it's not a tree.")
        return None, None

    origin_idx = origin_idx.item()
    origin = np.array([tree.x[origin_idx], tree.y[origin_idx], tree.z[origin_idx]])

    return origin, origin_idx


def fit_3d_ray(points):
    """Fit a ray through given points in 3D.

    This is done by minimizing the norm of the difference between v=(x, y, z)
    and the projection of v onto the ray

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) containing N points in 3D.

    Returns
    -------
    (theta, phi) or None:
        Spherical coordinates of the point on the unit sphere determining the
        ray direction. If the fitting fails then None is returned.
    """

    def residual(params, v):
        """Compute the distance between v and the normal constructed from params.

        Parameters
        ----------
        params : tuple
            The angles (theta, phi) of the normal vector in spherical coordinates.
        v : np.ndarray
            Array of shape (N, 3) representing N points in 3D.

        Returns
        -------
        dist
            The squared distance.
        """
        theta, phi = params
        normal = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )

        vv = np.sum(v * v, axis=1)
        vn = np.sum(v * normal, axis=1)

        dist = np.sqrt(np.clip(vv - vn**2, 0, None))

        return dist

    # construct initial parameters as a vector to the mean of all points
    mean = points.mean(axis=0)
    x0, y0, z0 = mean / np.linalg.norm(mean)
    theta0 = np.arccos(z0)
    phi0 = np.arctan2(y0, x0)
    params0 = np.array([theta0, phi0])

    # fit
    # The default value for `maxfev` (maximal number of function evaluations)
    # is 200 * len(x0) = 600. In some cases with a large numer of points (~2000)
    # this number of evaluations was too low for convergence. Trial and error
    # showed that increasing this number to the number of points in question
    # increases the chance of convergence with a negligible performance impact.
    x, cov_x, infodict, mesg, ier = scipy.optimize.leastsq(
        func=residual,
        x0=params0,
        args=points,
        full_output=True,
        maxfev=max(len(points), 600),
    )
    theta_fit, phi_fit = x

    # evaluate the output and return
    if ier not in (1, 2, 3, 4):
        logger.error(
            f"3D ray fitting failed, reason: {mesg}. "
            f"The returned values might be wrong."
        )

    return theta_fit, phi_fit


def fit_tree_ray(tree):
    """Fit a ray through the root of the tree.

    Parameters
    ----------
    tree : tmd.Tree.Tree.Tree)
        A TMD tree.

    Returns
    -------
    (theta, phi) or None:
        Spherical coordinates of the point on the unit sphere determining the
        ray direction. If the fitting fails then None is returned
    """
    origin, _ = get_tree_origin(tree)
    if origin is None:
        return None
    points = np.transpose([tree.x, tree.y, tree.z])

    return fit_3d_ray(points - origin)


def fit_tree_pca(tree):
    """Get the normalized tree principal component in spherical coordinates.

    Parameters
    ----------
    tree : tmd.Tree.Tree.Tree
        A TMD tree.

    Returns
    -------
    (theta, phi)
        The normalized principal component vector in spherical coordinates.
    """
    _, components, _ = get_tree_pca(tree)
    normal = components[0]
    normal = normal / np.linalg.norm(normal)
    x, y, z = normal
    theta = np.arccos(z)
    phi = np.arctan2(y, x)

    return theta, phi


def orient_neuron(orient_fn, neuron, in_place=False):
    """Orient the neuron apicals to point in the y-direction.

    Parameters
    ----------
    orient_fn : callable
        Function that finds the direction of a tree.
    neuron
        The neuron to orient.
    in_place : bool (optional)
        If true modify the given neuron instead of returning a copy.

    Returns
    -------
    new_neuron
        Oriented copy of the original neuron.
    """
    if in_place:
        new_neuron = neuron
    else:
        new_neuron = neuron.copy_neuron()

    if not hasattr(neuron, "apical") or len(neuron.apical) == 0:
        logger.warning("neuron has no apicals, no orienting done.")
        return new_neuron
    elif len(neuron.apical) > 1:
        logger.warning(
            "neuron has more than one apical; using the first apical for orientation",
        )

    theta, phi = orient_fn(neuron.apical[0])
    alpha = np.pi / 2 - phi
    beta = -(np.pi / 2 - theta)

    rot_z = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )
    rot_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(beta), -np.sin(beta)],
            [0, np.sin(beta), np.cos(beta)],
        ]
    )

    for tree in new_neuron.neurites:
        points = np.array([tree.x, tree.y, tree.z])
        tree.x, tree.y, tree.z = rot_x @ rot_z @ points

    return new_neuron


def orient_dataset(orient_fn, dataset_dir, output_dir=None, overwrite=False):
    """Orient all samples in a given dataset (not implemented yet).

    Parameters
    ----------
    orient_fn : callable
        The orientation function.
    dataset_dir : str or pathlib.Path
        The dataset directory.
    output_dir : str or pathlib.Path (optional)
        The output directory. If None an output directory name will be
        constructed by appending "_oriented" to the dataset directory name.
    overwrite : bool (optional)
        If true then the the output directory will be purged in case it already
        exists.

    Returns
    -------
    bool
        Whether or not the orientation was successful.
    """
    # TODO: implement orientation
    dataset_dir = pathlib.Path(dataset_dir)
    if not dataset_dir.exists():
        print("ERROR: the dataset directory doesn't exist", file=sys.stderr)
        return False

    # Purge / create output directory
    if output_dir is None:
        output_dir = dataset_dir.with_name(dataset_dir.name + "_oriented")
    else:
        output_dir = pathlib.Path(output_dir)

    if output_dir.exists() and overwrite:
        shutil.rmtree(str(output_dir))
    output_dir.mkdir(parents=True)

    # Read, orient, and write cells
    for layer in dataset_dir.iterdir():
        print(layer)

    return True
