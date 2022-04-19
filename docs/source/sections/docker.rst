Docker
======
We provide a docker file that allows you to run ``morphoclass`` on a docker
container. To build the docker image run the following command:

.. code-block:: sh

    docker build \
    -f docker/morphoclass.Dockerfile \
    -t morphoclass \
    --build-arg MORPHOCLASS_USERS \
    .

Notice the ``MORPHOCLASS_USERS`` build argument, it allows you to add custom
users to the container operating system. This way you'll be able to run your
container as a given user, which can be useful when writing files to a
bind-mounted volume. In this case you probably want to specify your own
username and user ID and you should set

.. code-block:: sh

    export MORPHOCLASS_USERS="$(whoami)/$(id -u)"

prior to running ``docker build``. You can also put this command into a ``.env``
file and source it. See the ``.env.example`` file for an example with multiple
custom users.

To run the container use the following command:

.. code-block:: sh

    docker run --rm -it  morphoclass

Note that the docker image sets up the environment for running ``morphoclass``,
but does not pre-install ``morphoclass`` itself. So probably the first thing you
want to do is to install the ``morphoclass`` package.

You might want to pass additional flags to ``docker run``:

* ``-v $PWD:/workdir``: bind-mounts the current working directory into
  ``/workdir``. If you have a local clone of ``morphoclass`` in the current
  working directory, then it will be available on the container under
  ``/workdir``, which can be useful if you want to install it from this clone
  or work on the code.
* ``-u $USER``: run the container as the current user. This is useful in
  combination with the ``-v`` flag above to ensure that all files you create on
  the mounted volume are assigned to your name. If you created custom users
  with a different name then simply replace ``$USER`` with the corresponding
  value.
* ``-p $PUBLIC_PORT:8888``: forward the port ``8888``, which is used by jupyter.
  You have to replace ``$PUBLIC_PORT`` by an available port number. This way
  you'll be able to run a ``jupyter lab`` instance directly on the container
  and open it in your browser.
* ``--gpus all``: if your machine has a CUDA-compatible graphics card you need
  to include this flag to make the GPUs available on the container.
* ``--name $MY_CONTAINER_NAME``: instead of assigning a random name to the
  container set a fixed one of your choice. Useful for long-running containers
  and for working in shared environments.

Just for reference, the ``docker run`` command with all flags included, the
external port set to ``35353``, and the container name to ``my-container``:

.. code-block:: sh

    docker run
    --rm \
    -it \
    -u $USER \
    -v $PWD:/workdir \
    -p 35353:8888 \
    --gpus all \
    --name my-container \
    morphoclass
