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
from __future__ import annotations

from click.testing import CliRunner

from morphoclass.console.cmd_preprocess_dataset import cli


def test_normal_run(mocker):
    preprocessor_cls = mocker.patch(
        "morphoclass.console.dataset_preprocessor.Preprocessor"
    )
    preprocessor = preprocessor_cls.return_value

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--dataset-type",
            "pyramidal-cells",
            "--morphologies-dir",
            "",
            "--db-file",
            "",
            "--output-csv-path",
            "",
            "--output-report-path",
            "",
        ],
    )
    assert result.exit_code == 0
    assert preprocessor.run.called_once()
    assert preprocessor.save_dataset_csv.called_once()
    assert preprocessor.save_report.called_once()
