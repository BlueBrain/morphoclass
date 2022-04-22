CLI
===

Overview
--------

The morphoclass package provides a command line interface (CLI) that allows
to train a deep learning model on morphological data, and to use a trained
model for inference. Moreover it provides some basic explain AI (XAI)
functionality that allows to reason about the models predictions.

The entry point for the CLI is called ``morphoclass`` and is automatically
installed together with the ``morphoclass`` package. You can test that
the installation worked correctly by running the following command:

.. code-block::

    $ deepmorphs -V
    deepmorphs, version 0.2.8.dev66


All functionality is organised into subcommands
``deepmorphs <sub-command> ...``.

There are some options to append to the command line:

.. code-block::

    Options:
      -h, --help       Show this message and exit.
      -v, --verbose    The logging verbosity: 0=WARNING (default), 1=INFO, 2=DEBUG
      --log-file PATH  Write a copy of all output to this file.
      -V, --version    Show the version and exit.

The most useful one is ``-v`` / ``-vv``, which enables a lot of useful output.
For interactive sessions I'd recommend to always use at least ``-v``, .e.g

.. code-block::

    $ deepmorphs -v <sub-command> ...

Subcommands
-----------

At the moment there are obsolete / duplicate commands due to ongoing
refactoring.

Currently used - done refactoring / being refactored:

* ``preprocess-dataset``  Preprocess a raw morphology dataset.
* ``organise-dataset``    Read a morphology dataset from a CSV file and
* ``plot-dataset-stats``  Ingest a morphology dataset through a CSV file and...
* ``train``               Train a morphology classification model.
* ``evaluate``            Evaluate a trained checkpoint.
* ``performance-table``   Generate a summary report of the performance of...
* ``morphometrics``       Run the morphometrics subcommand.

To refactor:

* ``explain-models``      XAI
* ``outlier-detection``   Outlier detection
* ``transfer-learning``   Transfer learning


Old / obsolete:

* ``train-and-evaluate``  (got split into ``train`` and ``evaluate``)
* ``performance-report``  replaced by ``evaluate`` and ``performance-table``

Please see the :doc:`dvc` section for more information on what the commands do.
