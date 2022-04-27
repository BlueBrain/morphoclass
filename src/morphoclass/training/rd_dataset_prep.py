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
"""Helper functions for preparation of morphology dataset splits."""
from __future__ import annotations

from morphoclass import transforms


def make_transform(
    dataset,
    feature_extractor=None,
    n_features=None,
    fitted_scaler=None,
    scaler_cls="FeatureRobustScaler",
    edge_weight=None,
):
    """Create a transform for a given dataset.

    The transform that is created is of the following form:

        MakeCopy('y', 'y_str', 'edge_index', 'tmd_neurites')
        FeatureExtractor
        Scaler
        EdgeWeight (optional)

    Parameters
    ----------
    dataset : morphoclass.data.MorphologyDataset
        A dataset.
    feature_extractor :
        A node feature extractor from `morphoclass.transforms`.
    n_features : int or list or tuple
        The number of features `feature_extractor` extracts. Must be
        provided if `feature_extractor` is not None.
    fitted_scaler :
        The feature scaler. If None then a new scaler will be fitted.
        Typical use case: use None for the training set to fit a new
        scaler, then re-used this scaler for the validation set.
    scaler_cls : str
        Only used of `fitted_scaler` is None. In this case a new scaler
        of this class will be fitted.
    edge_weight : int or None
        If not None then an additional edge weight transform will be
        included in the overall transform.

    Returns
    -------
    transform
        The overall transform.
    fitted_scaler
        The fitted scaler that is part of the overall transform. Useful
        when it need to be re-used, for example for a validation set.
    """
    if feature_extractor is None:
        feature_extractor = transforms.ExtractRadialDistances()
        n_features = 1
    else:
        if n_features is None:
            raise ValueError(
                "If feature_extractor is provided then also n_features must be set"
            )

    feature_indices = list(range(n_features))

    if fitted_scaler is None:
        old_transform = dataset.transform
        dataset.transform = transforms.Compose(
            [
                transforms.MakeCopy(),
                feature_extractor,
            ]
        )
        if scaler_cls == "FeatureRobustScaler":
            fitted_scaler = transforms.FeatureRobustScaler(
                feature_indices=feature_indices, with_centering=False
            )
            fitted_scaler.fit(dataset)
            assert fitted_scaler.center is None  # No shifting, root should stay at 0
            dataset.transform = old_transform
        elif scaler_cls == "FeatureMinMaxScaler":
            fitted_scaler = transforms.FeatureMinMaxScaler(
                feature_indices=feature_indices
            )
            fitted_scaler.fit(dataset)
            dataset.transform = old_transform
        else:
            dataset.transform = old_transform
            raise ValueError(f"Scaler class not supported: {scaler_cls}")

    transform_seq = [
        transforms.MakeCopy(
            keep_fields=["y", "y_str", "label", "edge_index", "tmd_neurites", "path"]
        ),
        feature_extractor,
        fitted_scaler,
    ]
    if edge_weight is not None:
        transform_seq.append(transforms.ExtractDistanceWeights(scale=edge_weight))
    transform = transforms.Compose(transform_seq)

    return transform, fitted_scaler


def prepare_rd_transforms(
    dataset_train,
    dataset_val=None,
    feature_extractor=None,
    n_features=None,
    scaler_cls="FeatureRobustScaler",
    edge_weight=None,
):
    """Prepare radial distance transforms for the given datasets.

    Parameters
    ----------
    dataset_train : morphoclass.data.MorphologyDataset
        The training set.
    dataset_val : morphoclass.data.MorphologyDataset, optional
        The validation set.
    feature_extractor
        A node feature extractor from `morphoclass.transforms`.
    n_features : int or list or tuple
        The number of features `feature_extractor` extracts. Must be
        provided if `feature_extractor` is not None.
    scaler_cls : str
        The name of the scaler class to use. Must be a scaler class available
        in `morphoclass.transforms.scalers`.
    edge_weight : int or None
        If not None then an additional edge weight transform will be
        included in the overall transform.

    Returns
    -------
    datasets
        If `dataset_val` is None the only the `dataset_train` is returned with
        the appropriate feature extractor attached to it. Otherwise a tuple
        with `(dataset_train, dataset_val)` is returned, with the `dataset_val`
        using a feature scaler that was fitted on the training data.
    """
    train_transform, fitted_scaler = make_transform(
        dataset=dataset_train,
        feature_extractor=feature_extractor,
        n_features=n_features,
        scaler_cls=scaler_cls,
        edge_weight=edge_weight,
    )
    dataset_train.transform = train_transform

    if dataset_val is None:
        return dataset_train
    else:
        val_transform, _ = make_transform(
            dataset=dataset_val,
            feature_extractor=feature_extractor,
            n_features=n_features,
            fitted_scaler=fitted_scaler,
            edge_weight=edge_weight,
        )
        dataset_val.transform = val_transform

        return dataset_train, dataset_val


def prepare_rd_split(
    dataset,
    train_idx,
    val_idx,
    feature_extractor=None,
    n_features=None,
    scaler_cls="FeatureRobustScaler",
    edge_weight=400,
):
    """Split a dataset into train/val sets and set up radial distance features.

    Parameters
    ----------
    dataset : morphoclass.data.MorphologyDataset
        The full dataset with train and validation data.
    train_idx : iterable
        The indices of the training-subset.
    val_idx : iterable
        The indices of the validation-subset.
    feature_extractor
        A node feature extractor from `morphoclass.transforms`.
    n_features : int or list or tuple
        The number of features `feature_extractor` extracts. Must be
        provided if `feature_extractor` is not None.
    scaler_cls : str
        The name of the scaler class to use. Must be a scaler class available
        in `morphoclass.transforms.scalers`.
    edge_weight : int or None
        If not None then an additional edge weight transform will be
        included in the overall transform.

    Returns
    -------
    dataset_train : morphoclass.data.MorphologyDataset
        The training-subset.
    dataset_val : morphoclass.data.MorphologyDataset
        The validation-subset.
    """
    dataset_train = dataset.index_select(train_idx)
    dataset_val = dataset.index_select(val_idx)

    return prepare_rd_transforms(
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        feature_extractor=feature_extractor,
        n_features=n_features,
        scaler_cls=scaler_cls,
        edge_weight=edge_weight,
    )


def prepare_smart_split(
    dataset, train_idx, val_idx, scaler_cls="FeatureRobustScaler", edge_weight=None
):
    """Prepare a dataset split given the indices.

    The split is prepared by constructing subsets using the train
    and validation indices, and by setting up transforms that
    extract the correct features.

    We have to distinguish between datasets with
    IPCs/HPSs and other datasets. The former have to use
    projections of the coordinates onto the y-axis as features,
    the latter the usual radial distances.

    In other words, L2 and L6 have to use projections, the layers
    L3, L4, and L5 radial distances.

    Note that it is assumed that the apicals are already correctly oriented,
    otherwise the projection features won't work correctly. The best way
    to do this is to include `transforms.OrientApicals` in the `pre_transform`
    of the `MorphologyDataset`

    Parameters
    ----------
    dataset : morphoclass.data.MorphologyDataset
        The dataset to draw the samples from.
    train_idx : list_like
        The indices for the training set.
    val_idx : list_like
        The indices for the validation set.
    scaler_cls : str, optional
        The name of the scaler class to be used for scaling features.
    edge_weight : int, optional
        The scale for the edge weight feature to be extracted.

    Returns
    -------
    dataset_train : MorphologyDataset
        The training subset of the dataset
    dataset_val : MorphologyDataset
        The validation subset of the dataset
    """
    # Determine layer and deduce the correct feature
    feature_extractor: transforms.ExtractDistances
    if dataset.guess_layer() in [2, 6]:
        feature_extractor = transforms.ExtractVerticalDistances()
    else:
        feature_extractor = transforms.ExtractRadialDistances()

    dataset_train, dataset_val = prepare_rd_split(
        dataset,
        train_idx,
        val_idx,
        feature_extractor=feature_extractor,
        n_features=1,
        scaler_cls=scaler_cls,
        edge_weight=edge_weight,
    )

    return dataset_train, dataset_val
