from __future__ import annotations

import json

import pytest

from morphoclass import transforms
from morphoclass.data import MorphologyDataset

# from morphoclass import training


@pytest.fixture(scope="session")
def dataset():
    dataset = MorphologyDataset.from_structured_dir(
        data_path="tests/data",
        layer="L5",
        pre_transform=transforms.ExtractTMDNeurites(neurite_type="apical"),
    )
    return dataset


@pytest.fixture(scope="session")
def tns_parameters_file(tmpdir_factory):
    file = tmpdir_factory.mktemp("data").join("parameters.json")
    params_tpc_a = {
        "apical": {
            "apical_distance": 0.0,
            "bias": 0.5,
            "bias_length": 250,
            "branching_method": "bio_oriented",
            "growth_method": "tmd_gradient",
            "metric": "path_distances",
            "modify": None,
            "orientation": [[0.0, 1.0, 0.0]],
            "radius": 0.3,
            "randomness": 0.24,
            "step_size": {"norm": {"mean": 1.0, "std": 0.2}},
            "targeting": 0.16,
            "tree_type": 4,
        },
        "axon": {},
        "basal": {
            "branching_method": "bio_oriented",
            "growth_method": "tmd",
            "metric": "path_distances",
            "modify": None,
            "orientation": None,
            "radius": 0.3,
            "randomness": 0.24,
            "step_size": {"norm": {"mean": 1.0, "std": 0.2}},
            "targeting": 0.16,
            "tree_type": 3,
        },
        "diameter_params": {"method": "M5"},
        "grow_types": ["apical", "basal"],
        "origin": [0.0, 0.0, 0.0],
    }
    params_tpc_b = {
        "apical": {
            "apical_distance": 0.0,
            "bias": 0.7,
            "bias_length": 350,
            "branching_method": "bio_oriented",
            "growth_method": "tmd_gradient",
            "metric": "path_distances",
            "modify": None,
            "orientation": [[0.0, 1.0, 0.0]],
            "radius": 0.3,
            "randomness": 0.26,
            "step_size": {"norm": {"mean": 1.0, "std": 0.2}},
            "targeting": 0.14,
            "tree_type": 4,
        },
        "axon": {},
        "basal": {
            "branching_method": "bio_oriented",
            "growth_method": "tmd",
            "metric": "path_distances",
            "modify": None,
            "orientation": None,
            "radius": 0.3,
            "randomness": 0.26,
            "step_size": {"norm": {"mean": 1.0, "std": 0.2}},
            "targeting": 0.14,
            "tree_type": 3,
        },
        "diameter_params": {"method": "M5"},
        "grow_types": ["apical", "basal"],
        "origin": [0.0, 0.0, 0.0],
    }
    parameters = {"L5_TPC_A": params_tpc_a, "L5_TPC_B": params_tpc_b}
    file.write(json.dumps(parameters))

    return file


# @pytest.mark.parametrize("ids", [None, [0, 1]], ids=["all", "subset"])
# def test_tns_distributions_from_dataset(dataset, ids):
#     distributions = training.tns_distributions_from_dataset(dataset, ids)
#
#     if ids is None:
#         expected_m_types = set(dataset.class_dict.values())
#     else:
#         expected_m_types = set(dataset.class_dict[dataset[i].y] for i in ids)
#
#     assert distributions.keys() == expected_m_types
#
#
# def test_read_tns_parameters(tns_parameters_file):
#     parameters = training.read_tns_parameters(tns_parameters_file)
#     assert parameters.keys() == {"L5_TPC_A", "L5_TPC_B"}
