.. _building_docs:

Building the Documentation
==========================
The documentation can be generated using ``sphinx``.
Follow the steps below to generate the documentation.

First make sure the stubs for the API documentation are up-to-date by
generating fresh ones.

.. code-block:: sh

    tox -e apidoc

Then use sphinx to build the documentation.

.. code-block:: sh

    make -C docs clean html
