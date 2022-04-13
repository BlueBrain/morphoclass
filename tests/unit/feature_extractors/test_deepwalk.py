from __future__ import annotations

import pathlib

import pytest

from morphoclass.data.morphology_embedding_data_loader import (
    MorphologyEmbeddingDataLoader,
)
from morphoclass.feature_extractors.interneurons import (
    feature_extractor as in_feature_extractor,
)
from morphoclass.feature_extractors.pyramidal_cells import (
    feature_extractor as pc_feature_extractor,
)

pytest.importorskip("deepwalk")


@pytest.mark.filterwarnings("ignore:Using or importing the ABCs:DeprecationWarning")
class TestDeepWalk:
    def test_feature_extractor_in(self):
        # files will be created
        dataset, loader, results = in_feature_extractor(
            file_name_suffix="deepwalk",
            input_csv="tests/data/L5_data.csv",
            embedding_type="deepwalk",
        )
        # check some outputs of deepwalk feature extraction
        assert len(dataset) == 4
        assert dataset[0].embedding.shape[1] == results["representation_size"]
        assert loader == MorphologyEmbeddingDataLoader
        assert results["class_dict"] == {
            0: "L5_TPC_A",
            1: "L5_TPC_B",
            2: "L5_TPC_C",
            3: "L5_UPC",
        }

        # files already exist
        dataset, loader, results = in_feature_extractor(
            file_name_suffix="deepwalk",
            input_csv="tests/data/L5_data.csv",
            embedding_type="deepwalk",
        )
        # check some outputs of deepwalk feature extraction
        assert len(dataset) == 4
        assert dataset[0].embedding.shape[1] == results["representation_size"]
        assert loader == MorphologyEmbeddingDataLoader
        assert results["class_dict"] == {
            0: "L5_TPC_A",
            1: "L5_TPC_B",
            2: "L5_TPC_C",
            3: "L5_UPC",
        }

        # remove all deepwalk files
        for d in dataset:
            if d.path:
                pathlib.Path(d.path).unlink()

        # csv with no data
        with pytest.raises(ValueError):
            dataset, loader, results = in_feature_extractor(
                file_name_suffix="deepwalk",
                input_csv="tests/data/L5_dummy.csv",
                embedding_type="deepwalk",
            )

        # not valid csv
        with pytest.raises(ValueError):
            dataset, loader, results = in_feature_extractor(
                file_name_suffix="deepwalk",
                input_csv="tests/data/L5_ghost.csv",
                embedding_type="deepwalk",
            )

    def test_feature_extractor_pc(self):
        # files will be created
        dataset, loader, results = pc_feature_extractor(
            file_name_suffix="deepwalk",
            input_csv="tests/data/L5_data.csv",
            embedding_type="deepwalk",
        )
        # check some outputs of deepwalk feature extraction
        assert len(dataset) == 4
        assert dataset[0].embedding.shape[1] == results["representation_size"]
        assert loader == MorphologyEmbeddingDataLoader
        assert results["class_dict"] == {
            0: "L5_TPC_A",
            1: "L5_TPC_B",
            2: "L5_TPC_C",
            3: "L5_UPC",
        }

        # files already exist
        dataset, loader, results = pc_feature_extractor(
            file_name_suffix="deepwalk",
            input_csv="tests/data/L5_data.csv",
            embedding_type="deepwalk",
        )
        # check some outputs of deepwalk feature extraction
        assert len(dataset) == 4
        assert dataset[0].embedding.shape[1] == results["representation_size"]
        assert loader == MorphologyEmbeddingDataLoader
        assert results["class_dict"] == {
            0: "L5_TPC_A",
            1: "L5_TPC_B",
            2: "L5_TPC_C",
            3: "L5_UPC",
        }

        # remove all deepwalk files
        for d in dataset:
            if d.path:
                pathlib.Path(d.path).unlink()

        # L6 and L2 are handled differently
        dataset, loader, results = pc_feature_extractor(
            file_name_suffix="deepwalk",
            input_csv="tests/data/L6_data.csv",
            embedding_type="deepwalk",
        )
        # check some outputs of deepwalk feature extraction
        assert len(dataset) == 1
        assert dataset[0].embedding.shape[1] == results["representation_size"]
        assert loader == MorphologyEmbeddingDataLoader
        assert results["class_dict"] == {
            0: "L6_BPC",
        }
        # remove all deepwalk files
        for d in dataset:
            if d.path:
                pathlib.Path(d.path).unlink()

        # csv with no data
        with pytest.raises(ValueError):
            dataset, loader, results = pc_feature_extractor(
                file_name_suffix="deepwalk",
                input_csv="tests/data/L5_dummy.csv",
                embedding_type="deepwalk",
            )

        # not valid csv
        with pytest.raises(ValueError):
            dataset, loader, results = pc_feature_extractor(
                file_name_suffix="deepwalk",
                input_csv="tests/data/L5_ghost.csv",
                embedding_type="deepwalk",
            )
