Project Tree
============

Files one should see after cloning:

.. code-block::

    ├── docker/
    ├── docs/
    ├── dvc/
    ├── src/
    ├── tests/
    ├── .dockerignore
    ├── .gitignore
    ├── .pre-commit-config.yaml
    ├── AUTHORS.md
    ├── constraints.txt
    ├── CONTRIBUTING.md
    ├── install.sh
    ├── LICENSE.txt
    ├── pyproject.toml
    ├── README.md
    ├── requirements.txt
    ├── setup.cfg
    └── tox.ini

Folders:

* ``docker/``: files for building the MorphoClass docker image
* ``docs/``: sphinx documentation
* ``dvc/``: main data and experiments folder. When working with DVC/data always
  run ``cd dvc`` beforehand.
* ``src/``: the package source files
* ``tests/``: unit tests

Files:

* ``.dockerignore``: files to not send to the docker daemon when building the
  docker image (make sure to ignore big files!)
* ``.gitignore``: the files to not track with git
* ``.pre-commit-config.yaml``: configuration of pre-commit hooks, activate via
  ``pre-commit install``
* ``AUTHORS.md``: The authors of MorphoClass
* ``constraints.txt``: version constraints for third-party dependencies, to be
  used with ``pip install -c constraints.txt ...``
* ``CONTRIBUTING.md``: guidelines for contributions to the MorphoClass package
* ``install.sh``: The installation script, see the
  :doc:`installation` section for more details
  for details
* ``LICENSE.txt``: The MorphoClass license
* ``pyproject.toml``: standard packaging configuration. Also contains
  configuration for ``mypy``
* ``README.md``: the main README file
* ``requirements.txt``: the project requirements with pinned versions; useful
  for development purposes
* ``setup.cfg``: standard packaging configuration
* ``tox.ini``: tox configuration, a testing automation tool
