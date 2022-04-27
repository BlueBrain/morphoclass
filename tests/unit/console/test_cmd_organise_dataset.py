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

import pandas as pd
from click.testing import CliRunner

from morphoclass.console.cmd_organise_dataset import cli


def test_normal_run(tmp_path):
    # Prepare test data
    csv_path = tmp_path / "dataset.csv"
    morph_dir = tmp_path / "morphologies"
    morph_dir.mkdir()
    morph_1_path = morph_dir / "m1.h5"
    morph_2_path = morph_dir / "m2.h5"
    morph_1_path.touch()
    morph_2_path.touch()
    dataset = pd.DataFrame(
        [
            {
                "morph_path": morph_1_path,
                "mtype": "L1_A",
            },
            {
                "morph_path": morph_2_path,
                "mtype": "L1_B",
            },
        ]
    )
    dataset.to_csv(csv_path, index=False)
    output_dir = tmp_path / "output"

    # Run test
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--input-csv-path", csv_path, "--output-dataset-directory", output_dir]
    )
    assert result.exit_code == 0
    assert output_dir.exists()
    assert (output_dir / "dataset.csv").exists()
    assert (output_dir / "L1" / "A" / "m1.h5").exists()
    assert (output_dir / "L1" / "B" / "m2.h5").exists()
    assert (output_dir / "L1" / "dataset.csv").exists()
