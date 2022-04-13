from __future__ import annotations

import logging

import pytest
from click.testing import CliRunner

from morphoclass.console import main


class TestConfigureLogging:
    @pytest.mark.parametrize(
        "log_level",
        (logging.DEBUG, logging.INFO, logging.WARNING),
    )
    def test_setting_log_level(self, log_level, caplog):
        main._configure_logging(log_level, None)
        logger = logging.getLogger("morphoclass")
        logger.propagate = True  # otherwise capsys doesn't work
        with caplog.at_level(log_level, "morphoclass"):
            logger.log(log_level, "test message")

        assert "test message" in caplog.text

    def test_logging_to_file(self, caplog, tmp_path):
        log_file = tmp_path / "test.log"
        main._configure_logging(logging.DEBUG, log_file)
        logger = logging.getLogger("morphoclass")
        logger.debug("test message")
        assert log_file.exists()
        with log_file.open() as fh:
            log = fh.read()
        assert "test message" in log


class TestMainEntrypoint:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main.cli, ["--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize(
        ("verbosity_flag", "log_level"),
        (
            ("", logging.WARNING),
            ("-v", logging.INFO),
            ("-vv", logging.DEBUG),
            ("-vvv", logging.DEBUG),
        ),
    )
    def test_verbosity(self, verbosity_flag, log_level):
        @main.cli.command(name="sub")
        def subcommand():
            logger = logging.getLogger("morphoclass")
            assert logger.level == log_level

        runner = CliRunner()
        params = ["sub"]
        if verbosity_flag:
            params = [verbosity_flag, *params]
        result = runner.invoke(main.cli, params)
        assert result.exit_code == 0
