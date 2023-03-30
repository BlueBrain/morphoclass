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
"""The main morphoclass CLI entrypoint."""
from __future__ import annotations

import logging
import pathlib

import click

import morphoclass
from morphoclass.console import cmd_evaluate
from morphoclass.console import cmd_extract_features
from morphoclass.console import cmd_extract_features_and_predict
from morphoclass.console import cmd_morphometrics
from morphoclass.console import cmd_organise_dataset
from morphoclass.console import cmd_performance_table
from morphoclass.console import cmd_plot_dataset_stats
from morphoclass.console import cmd_predict
from morphoclass.console import cmd_preprocess_dataset
from morphoclass.console import cmd_train
from morphoclass.console import cmd_xai
from morphoclass.console import outlier_detection
from morphoclass.console import transfer_learning

logger = logging.getLogger(__name__)


def _configure_logging(level: int, log_file_path: pathlib.Path | None) -> None:
    """Configure logging for the CLI application.

    Logging is the primary means of emitting messages of morphoclass, both in
    the CLI part and in the rest of the library. Therefore, it's important we
    configure it in the main CLI entrypoint so that it's possible to emit
    all relevant messages.

    We configure the top logger for "morphoclass" so that all our messages
    from the CLI modules as well as from the rest of the library are
    handled. At the same time the logging of third-party modules won't
    be captured.

    We also configure the "morphoclass" logger to not propagate log records
    to the root logger, so it's still possible to configure the root logger
    to handle third-party logging.

    Parameters
    ----------
    level
        The logging level.
    log_file_path
        The output file for log messages. This is in addition to the on-screen
        outputs in the terminal. Whatever this value, the terminal output
        will always be configured. A value of None means no log file output.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s (%(levelname).1s) %(message)s",
        datefmt="%H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    top_module, _period, _rest = __name__.partition(".")
    morphoclass_logger = logging.getLogger(top_module)
    morphoclass_logger.setLevel(level)
    morphoclass_logger.addHandler(handler)
    morphoclass_logger.propagate = False

    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        morphoclass_logger.addHandler(file_handler)


@click.group(
    help="""
    Welcome to the command line application for morphoclass.

    All functionality is provided through respective sub-commands.
    To learn more about their functionality call the corresponding
    sub-command with the --help flag to see a detailed description.
    """
)
@click.help_option("-h", "--help")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="The logging verbosity: 0=WARNING (default), 1=INFO, 2=DEBUG",
)
@click.option(
    "--log-file",
    "log_file_path",
    type=pathlib.Path,
    help="Write a copy of all output to this file.",
)
@click.version_option(morphoclass.__version__, "-V", "--version")
def cli(verbose: int, log_file_path: pathlib.Path | None) -> None:
    """Run the command line interface for morphoclass.

    For detailed instructions see the documentation of the
    corresponding sub-commands.

    Parameters
    ----------
    verbose
        The output verbosity level of the CLI. The CLI output is produced
        through logging, and therefore this parameter controls the logging
        level. Verbosity value 2 and above corresponds to the logging level
        DEBUG, the value 1 to INFO, 0 and below to WARNING.
    log_file_path
        The log file for mirroring the logging outputs from the screen output.
        Note that only log messages coming from morphoclass will be logged,
        but not those from third-party modules.
    """
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    _configure_logging(level, log_file_path)
    logger.info("Running them morphoclass entrypoint")


cli.add_command(transfer_learning.cli)
cli.add_command(outlier_detection.cli)
cli.add_command(cmd_xai.cli)
cli.add_command(cmd_organise_dataset.cli)
cli.add_command(cmd_plot_dataset_stats.cli)
cli.add_command(cmd_predict.cli)
cli.add_command(cmd_preprocess_dataset.cli)
cli.add_command(cmd_train.cli)
cli.add_command(cmd_evaluate.cli)
cli.add_command(cmd_performance_table.cli)
cli.add_command(cmd_extract_features.cli)
cli.add_command(cmd_extract_features_and_predict.cli)
cli.add_command(cmd_morphometrics.cli)
