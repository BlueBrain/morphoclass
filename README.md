# Morphology-Classification
MorphoClass is a toolbox for neuronal morphologies processing and
classification using machine learning.

## Installation
Clone the repository and set up the virtual environment
```sh
git clone git@github.com:BlueBrain/morphoclass.git
cd morphoclass
python --version  # should be 3.8
python -m venv venv
. venv/bin/activate
```

Install `morphoclass`
```sh
./install.sh
```

## Installation on the BB5
Allocate a node
```sh
ssh bbpv1
salloc -N1 -A<projXXX> -t 2:00:00
```

Clone the repository and set up the virtual environment
```sh
git clone git@github.com:BlueBrain/morphoclass.git
cd morphoclass
module load unstable
module load python
python --version  # should be 3.8
python -m venv venv
. venv/bin/activate
```

Install `morphoclass`
```sh
./install.sh
```

## Known Installation Issues
In some cases the installation might fail because either `cmake` or `hdf5` are
missing. On macOS these can be installed using `brew`:
`cmake` and `hdf5` via
```sh
brew install cmake hdf5
```

In some instances an incompatibility of certain compilers was reported.
Installing the following conda packages resolved the problem:
```sh
conda install clang_osx-64 clangxx_osx-64 gfortran_osx-64
```

## Documentation
The documentation can be generated using `sphinx`.
Follow the steps below to generate the documentation.

First make sure the stubs for the API documentation are up-to-date by
generating fresh ones.
```sh
tox -e apidoc
```

Then use sphinx to build the documentation.
```sh
make -C docs clean html
```

Open the file `docs/build/html/index.html` to view the documentation.

## Docker
We provide a docker file that allows you to run `morphoclass` on a docker
container. To build the docker image run the following command:
```sh
docker build \
    -f docker/morphoclass.Dockerfile \
    -t morphoclass \
    --build-arg MORPHOCLASS_USERS \
    .
```
Notice the `MORPHOCLASS_USERS` build argument, it allows you to add custom
users to the container operating system. This way you'll be able to run your
container as a given user, which can be useful when writing files to a
bind-mounted volume. In this case you probably want to specify your own
username and user ID and you should set
```sh
export MORPHOCLASS_USERS="$(whoami)/$(id -u)"
```
prior to running `docker build`. You can also put this command into a `.env`
file and source it. See the `.env.example` file for an example with multiple
custom users.

To run the container use the following command:
```sh
docker run --rm -it  morphoclass
```
Note that the docker image sets up the environment for running `morphoclass`,
but does not pre-install `morphoclass` itself. So probably the first thing you
want to do is to install the `morphoclass` package.

You might want to pass additional flags to `docker run`:
* `-v $PWD:/workdir`: bind-mounts the current working directory into `/workdir`.
  If you have a local clone of `morphoclass` in the current working directory,
  then it will be available on the container under `/workdir`, which can be
  useful if you want to install it from this clone or work on the code.
* `-u $USER`: run the container as the current user. This is useful in
  combination with the `-v` flag above to ensure that all files you create on
  the mounted volume are assigned to your name. If you created custom users
  with a different name then simply replace `$USER` with the corresponding
  value.
* `-p $PUBLIC_PORT:8888`: forward the port `8888`, which is used by jupyter.
  You have to replace `$PUBLIC_PORT` by an available port number. This way
  you'll be able to run a `jupyter lab` instance directly on the container and
  open it in your browser.
* `--gpus all`: if your machine has a CUDA-compatible graphics card you need to
  include this flag to make the GPUs available on the container.
* `--name $MY_CONTAINER_NAME`: instead of assigning a random name to the
  container set a fixed one of your choice. Useful for long-running containers
  and for working in shared environments.

Just for reference, the `docker run` command with all flags included, the
external port set to `35353`, and the container name to `my-container`:
```sh
docker run 
    -v $PWD:/workdir \
    -p 35353:8888 \
    --rm \
    -it \
    --gpus=all \
    -u $USER \
    --name my-container \
    morphoclass
```
