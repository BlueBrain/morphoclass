"""Script to collect all models' performances."""
from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence

import click

logger = logging.getLogger(__name__)


@click.command(
    name="performance-table",
    help="""
    Generate a summary report of the performance of selected trained models.

    This command will load all model checkpoints from the provided
    checkpoint directory and compile their performance metrics into
    an HTML report.
    """,
)
@click.argument(
    "checkpoint_paths",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(exists=False, file_okay=False),
    help="The HTML report output directly.",
)
def cli(
    checkpoint_paths: Sequence[str | pathlib.Path], output_dir: str | pathlib.Path
) -> None:
    """Compile the table with models' results.

    Parameters
    ----------
    checkpoint_paths
        All checkpoint files
    output_dir
        The HTML report output directly.
    """
    logger.info(f"Got {len(checkpoint_paths)} checkpoints")
    logger.info("Loading libraries")
    from morphoclass.console.performance_table import make_performance_table

    logger.info("Creating the performance summary table")
    make_performance_table(checkpoint_paths, output_dir)
