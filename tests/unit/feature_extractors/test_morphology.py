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
from __future__ import annotations

import pytest

from morphoclass.data.morphology_data_loader import MorphologyDataLoader
from morphoclass.feature_extractors.interneurons import (
    feature_extractor as in_feature_extractor,
)
from morphoclass.feature_extractors.pyramidal_cells import (
    feature_extractor as pc_feature_extractor,
)


class TestMorphology:
    def test_feature_extractor_in(self):
        dataset, loader, results = in_feature_extractor(
            file_name_suffix=None,
            input_csv="tests/data/L5_data.csv",
            embedding_type="morphology",
        )
        # check some outputs of morphology feature extraction
        assert len(dataset) == 4
        assert loader == MorphologyDataLoader
        assert results["class_dict"] == {
            0: "L5_TPC_A",
            1: "L5_TPC_B",
            2: "L5_TPC_C",
            3: "L5_UPC",
        }

        # csv with no data
        with pytest.raises(ValueError):
            dataset, loader, results = in_feature_extractor(
                file_name_suffix=None,
                input_csv="tests/data/L5_dummy.csv",
                embedding_type="morphology",
            )

        # not valid csv
        with pytest.raises(ValueError):
            dataset, loader, results = in_feature_extractor(
                file_name_suffix=None,
                input_csv="tests/data/L5_ghost.csv",
                embedding_type="morphology",
            )

    def test_feature_extractor_pc(self):
        dataset, loader, results = pc_feature_extractor(
            file_name_suffix=None,
            input_csv="tests/data/L5_data.csv",
            embedding_type="morphology",
        )
        # check some outputs of morphology feature extraction
        assert len(dataset) == 4
        assert loader == MorphologyDataLoader
        assert results["class_dict"] == {
            0: "L5_TPC_A",
            1: "L5_TPC_B",
            2: "L5_TPC_C",
            3: "L5_UPC",
        }

        # L6 and L2 are handled differently
        dataset, loader, results = pc_feature_extractor(
            file_name_suffix=None,
            input_csv="tests/data/L6_data.csv",
            embedding_type="morphology",
        )
        # check some outputs of morphology feature extraction
        assert len(dataset) == 1
        assert loader == MorphologyDataLoader
        assert results["class_dict"] == {
            0: "L6_BPC",
        }

        # csv with no data
        with pytest.raises(ValueError):
            dataset, loader, results = pc_feature_extractor(
                file_name_suffix=None,
                input_csv="tests/data/L5_dummy.csv",
                embedding_type="morphology",
            )

        # not valid csv
        with pytest.raises(ValueError):
            dataset, loader, results = pc_feature_extractor(
                file_name_suffix=None,
                input_csv="tests/data/L5_ghost.csv",
                embedding_type="morphology",
            )
