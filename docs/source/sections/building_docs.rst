Building the Documentation
==========================

The documentation can be generated using ``sphinx``.
Follow the steps below to generate the documentation.

To build the docs we will need some extra dependencies, so please run the
following.

.. code-block:: sh

    pip install -e ".[docs]"

Now, first of all make sure the stubs for the API documentation are up-to-date
by generating fresh ones.

.. code-block:: sh

    tox -e apidoc

Finally, use sphinx to build the documentation.

.. code-block:: sh

    make -C docs clean html

You can now open the file ``docs/build/html/index.html`` to view the
documentation.
