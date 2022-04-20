Installation
============

Basic
-----
This is the basic installation described in the
`README <https://github.com/BlueBrain/morphoclass#readme>`__ file. It is
suitable for non-development purposes, e.g. using the CLI and the python
API.

Clone the repository and set up the virtual environment

.. code-block:: sh

    git clone git@github.com:BlueBrain/morphoclass.git
    cd morphoclass
    python --version  # should be 3.8
    python -m venv venv
    . venv/bin/activate

Install ``morphoclass``

.. code-block:: sh

    ./install.sh


Check the Installation
----------------------
Test if the python API works by running the following command:

.. code-block:: sh

    python -c 'import morphoclass; print(morphoclass.__version__)'

The should print the currently installed MorphoClass version in the SemVer
format ``x.x.x``.

To test the CLI, run the main entrypoint in the terminal:

.. code-block:: sh

    morphoclass --version

This should print something like ``morphoclass, version x.x.x``.

Development
-----------
The basic approach described above is in theory also suitable for development,
except that it fails to do three things:

* Installing the ``dev`` and ``docs`` extras
* Pinning the versions of all dependencies as specified in ``requirements.txt``
* The installation is not in editable mode

To fix this, instead of just running ``./install.sh`` above please run

.. code-block:: sh

    pip install -r requirements.txt -c constraints.txt
    ./install.sh
    pip install -e '.[dev,docs]'

If you're planning to contribute to the code you should also activate
``pre-commit`` before making your first commit:

.. code-block:: sh

    pre-commit install

Why an Installation Script?
---------------------------
You may wonder why an installation script is required. The root of the problem
is the fact that two packages that MorphoClass depends on, ``torch`` and
``torch-geometric`` come in different flavours and need to be installed
differently depending on the operating system, the presence of a GPU, etc. To
get an overview on the installation options for these two packages please
see their corresponding docs:

* https://pytorch.org/get-started/locally
* https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Because of this the installation happens in three different steps

1. Installing the ``morphoclass`` package as usual
2. Installing/re-installing ``torch`` manually
3. Installing ``torch-geometric`` manually

You can follow the instructions below to install ``morphoclass`` from scratch
using these three steps. This installation will be suitable for development.

1. Install the ``morphoclass`` package as usual
...............................................
The first steps are exactly identical to what was described in the previous
sections.

Clone the repository and set up a virtual environment with ``python3.8``:

.. code-block:: sh

    git clone git@github.com:BlueBrain/morphoclass.git
    cd morphoclass
    python --version  # should be 3.8
    python -m venv venv
    . venv/bin/activate

Install the requirements:

.. code-block:: sh

    pip install -r requirements.txt -c constraints.txt

Install ``morphoclass`` in editable mode and with the ``dev`` and ``docs``
extras:

.. code-block::

    pip install -e '.[dev,docs]'

Activate ``pre-commit``:

.. code-block:: sh

    pre-commit install

2. Install/re-install ``torch`` manually
........................................
Normally ``torch`` is already included in ``requirements.txt`` and
``setup.cfg``. In many cases it just works out of the box.

When running ``pip install torch`` it will

* On macOS: install the CPU version (GPUs don't work on macs)
* On linux: install the CUDA 10.2 version

Sometimes this causes issues on linux machines, in particular non-GPU nodes on
the BB5. In this case the CPU flavour has to be forced via

.. code-block:: sh

    pip install \
    "torch==1.7.1+cpu" \
    --find-links "https://download.pytorch.org/whl/cpu/torch_stable.html"

* Make sure the torch version matches that in ``requirements.txt``
* Check the PyTorch website for more details
* The ``install.sh`` script does heuristics to detect the problematic situation
  and to force the CPU flavour if necessary

3. Install ``torch-geometric`` manually
.......................................
There are several helper packages that come with ``torch-geometric``:

* Main package: ``torch-geometric``
* Helper packages:

  * ``torch-scatter``
  * ``torch-sparse``
  * ``torch-cluster``
  * ``torch-spline-conv``

Problem: need to pick the right flavour to match the version of ``torch``.
Here are the commands that do all that automatically assuming the ``torch``
version is ``1.7.1`` (check ``requirements.txt`` for the correct version):

.. code-block:: sh

    CUDA="cpu"
    TORCH="1.7.1"
    pip install \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    --no-index \
    --find-links "https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html"

* The ``install.sh`` scripts does all that
* We're still using an older version of PyTorch-Geometric
* The latest versions of PyTorch Geometric might have improved the installation

At this point you should have a complete development installation of
MorphoClass.

DeepWalk and CleanLab
---------------------
Some parts of MorphoClass's code make reference to ``deepwalk`` and
``cleanlab``. However, the corresponding packages are not installed as part
of MorphoClass's installation because of licensing issues. Please install
them manually if you wish to use the functionality related to them.

To check if ``deepwalk`` is installed and to get instructions on how to
install it in case it is not please run the following command:

.. code-block:: sh

    $ python -c 'from morphoclass import deepwalk; deepwalk.warn_if_not_installed()'

To check if ``cleanlab`` is installed and to get instructions on how to
install it in case it is not please run the following command:

.. code-block:: sh

    $ python -c 'from morphoclass import cleanlab; cleanlab.warn_if_not_installed()'
