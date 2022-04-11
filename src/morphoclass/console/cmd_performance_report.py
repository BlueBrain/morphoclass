"""Script to collect all models' performances."""
from __future__ import annotations

import logging
import pathlib

import click

logger = logging.getLogger(__name__)


@click.command(
    name="performance-report",
    help="""
    Generate a summary report about the performance of all trained models.

    This command will load all model checkpoints from the provided
    checkpoint directory and compile their performance metrics into
    an HTML report.
    """,
)
@click.help_option("-h", "--help")
@click.option(
    "--checkpoint-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="The directory with all checkpoint files.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(exists=False, file_okay=False),
    help="The HTML report output directly.",
)
def cli(checkpoint_dir: str | pathlib.Path, output_dir: str | pathlib.Path) -> None:
    """Compile the table with models' results.

    Parameters
    ----------
    checkpoint_dir
        The directory with all checkpoint files.
    output_dir
        The HTML report output directly.
    """
    from morphoclass.console.performance_report import make_performance_report

    make_performance_report(checkpoint_dir, output_dir)
