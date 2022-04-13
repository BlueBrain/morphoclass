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
