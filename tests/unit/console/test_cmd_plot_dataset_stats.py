from __future__ import annotations

from click.testing import CliRunner

from morphoclass.console.cmd_plot_dataset_stats import cli


def test_normal_run(mocker):
    plotter_cls = mocker.patch("morphoclass.console.stats_plotter.DatasetStatsPlotter")
    plotter = plotter_cls.return_value

    runner = CliRunner()
    result = runner.invoke(cli, ["--input-csv-path", "", "--output-report-path", ""])
    assert result.exit_code == 0
    assert plotter.run.called_once()
    assert plotter.save_report.called_once()
