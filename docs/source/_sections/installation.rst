Installation
============

The code for MorphoClass is hosted on GitHub repository and can be found here:
`<https://github.com/BlueBrain/morphoclass>`_.
To install it first clone the repository in change into the directory just cloned:

.. code-block::

    $ git clone https://bbpgitlab.epfl.ch/ml/morphoclass
    $ cd morphoclass

Then install with pip:

.. code-block::

    $ export PIP_INDEX_URL="https://bbpteam.epfl.ch/repository/devpi/simple"
    $ pip install .

The relevant files have now been copied to the python installation directory and
therefore the source code files may be deleted.

In case you want to alter the code and have the changes applied to your installation you may
want to install the package in editable mode:

.. code-block::

    $ pip install  -e .

Note, however, that the installation now links to the current directory and one should not
delete it, otherwise the installation will be broken.
