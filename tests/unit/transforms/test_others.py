from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch
from torch_geometric.transforms import Compose

from morphoclass.data import MorphologyDataset
from morphoclass.transforms import AddOneHotLabels
from morphoclass.transforms import AverageBranchOrder
from morphoclass.transforms import ExtractCoordinates
from morphoclass.transforms import ExtractTMDNeurites
from morphoclass.transforms import ExtractTMDNeuron
from morphoclass.transforms import GlobalFeatureToLabel
from morphoclass.transforms import MakeCopy
from morphoclass.transforms import ZeroOutFeatures


@pytest.fixture
def dataset():
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeurites(neurite_type="apical"),
    )
    return dataset


def test_add_one_hot_labels(dataset):
    transform = AddOneHotLabels(fn_get_oh_label=lambda n: np.eye(10)[n])
    assert str(transform).startswith("AddOneHotLabels(fn_get_oh_label=")

    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", transform=transform
    )

    for sample in dataset:
        assert sample.y_oh.size() == (1, 10)

    # Test missing y argument in Data
    for sample in dataset:
        delattr(sample, "y")

    for i in range(len(dataset)):
        with pytest.raises(ValueError):
            dataset.__getitem__(i)


def test_extract_tmd_neurites(capsys):
    assert (
        str(ExtractTMDNeurites(neurite_type="apical", from_tmd_neuron=False))
        == "ExtractTMDNeurites(neurite_type=apical,from_tmd_neuron=False)"
    )

    dataset = MorphologyDataset.from_structured_dir(data_path="tests/data", layer="L5")

    # Extracting from MorphIO morphology
    dataset.transform = ExtractTMDNeurites(neurite_type="apical")
    data = dataset[0]
    assert hasattr(data, "tmd_neurites")
    delattr(data, "tmd_neurites")

    # Extracting from TMD morphology, but TMD neuron missing
    dataset.transform = ExtractTMDNeurites(neurite_type="apical", from_tmd_neuron=True)
    with pytest.raises(ValueError):
        dataset.__getitem__(0)

    # Extracting from TMD morphology
    dataset.transform = Compose(
        [
            ExtractTMDNeuron(),
            ExtractTMDNeurites(neurite_type="apical", from_tmd_neuron=True),
        ]
    )
    data = dataset[0]
    assert hasattr(data, "tmd_neurites")

    # Issue warning if neuron has no apicals
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=ExtractTMDNeuron(),
        transform=ExtractTMDNeurites(neurite_type="apical", from_tmd_neuron=True),
    )
    dataset[0].tmd_neuron.apical = []
    data = dataset[0]
    _, err = capsys.readouterr()
    neuron_name = pathlib.Path(data.path).stem.strip()
    assert err.strip() == f"WARNING: Neuron {neuron_name} has no apical"
    delattr(dataset.data[0], "path")
    _ = dataset[0]
    _, err = capsys.readouterr()
    assert err.strip() == "WARNING: Neuron has no apical"

    # Test no `morphology` attribute
    for data in dataset:
        delattr(data, "morphology")
    dataset.transform = ExtractTMDNeurites(neurite_type="apical")
    with pytest.raises(ValueError):
        dataset.__getitem__(0)


def test_extract_tmd_neuron():
    transform = ExtractTMDNeuron()
    assert str(transform) == "ExtractTMDNeuron()"

    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=transform
    )
    data = dataset[0]
    assert hasattr(data, "tmd_neuron")

    for data in dataset:
        delattr(data, "morphology")
    dataset.transform = transform
    with pytest.raises(ValueError):
        dataset.__getitem__(0)


def test_make_copy():
    dataset = MorphologyDataset.from_structured_dir(data_path="tests/data", layer="L5")

    assert dataset[0] is dataset.data[0]

    dataset.transform = MakeCopy()

    assert dataset[0] is not dataset.data[0]


def test_zero_out_features():
    pre_transform = Compose(
        [
            ExtractTMDNeurites(neurite_type="apical"),
            ExtractCoordinates(),
        ]
    )
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data", layer="L5", pre_transform=pre_transform
    )

    for sample in dataset:
        assert not torch.all(sample.x == 0)

    dataset.transform = Compose(
        [
            MakeCopy(keep_fields="x"),
            ZeroOutFeatures(),
        ]
    )

    for sample in dataset:
        assert sample.x.shape[-1] == 3  # features = coordinates
        assert torch.all(sample.x == 0)


def test_global_feature_to_label(dataset):
    dataset.transform = Compose(
        [AverageBranchOrder(), GlobalFeatureToLabel(global_feature_index=0)]
    )

    assert all(sample.u[0, 0].item() == sample.y for sample in dataset)

    from morphoclass.data import MorphologyDataLoader

    loader = MorphologyDataLoader(dataset, batch_size=2)
    next(iter(loader))
