"""XAI subcommands."""
from __future__ import annotations

import pathlib

import click


@click.command(name="explain-models", help="Create an XAI report.")
@click.option(
    "--results-file",
    default="params.yaml",
    required=True,
    type=click.Path(dir_okay=False),
    help="The HTML report output path.",
)
@click.option(
    "--checkpoint-path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Path to a model checkpoint.",
)
def cli(results_file: str | pathlib.Path, checkpoint_path: str | pathlib.Path) -> None:
    """Create an XAI report.

    Parameters
    ----------
    results_file
        The HTML report output path.
    checkpoint_path
        Path to a model checkpoint.
    """
    import torch

    from morphoclass.xai.reports.xai_report import xai_report

    checkpoint = torch.load(checkpoint_path)
    dataset_name = checkpoint["dataset_name"]
    feature_extractor_name = checkpoint["feature_extractor_name"]
    input_csv = checkpoint["input_csv"]
    model_class = checkpoint["model_class"]
    model_params = checkpoint["model_params"]
    model_old = checkpoint["all"]["model"]
    seed = checkpoint["seed"]

    xai_report(
        results_file,
        dataset_name,
        feature_extractor_name,
        input_csv,
        model_class,
        model_params,
        model_old,
        seed=seed,
    )
