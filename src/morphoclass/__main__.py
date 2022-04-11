"""The main entry point for the morphoclass module.

This allows to run the morphoclass module via ``python -m morphoclass`` and
will trigger the main CLI entry point. This should be equivalent to
running the entry point directly via ``morphoclass``. The added value is that
the CLI entry point can be started through the debugger via
``python -m pdb -m morphoclass``.
"""
from __future__ import annotations

from morphoclass.console.main import cli

cli()
