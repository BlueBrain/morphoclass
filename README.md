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

# Examples

MorphoClass functionalities can be accessed using either a command line
interface or by importing `morphoclass` as a Python module and using it as a
library.

## Command line interface

By installing MorphoClass, we get access to a command line interface with a
main entrypoint `morphoclass` and various sub-commands.
```shell
$ morphoclass --help
Usage: morphoclass [OPTIONS] COMMAND [ARGS]...

  Welcome to the command line application for morphoclass.

  All functionality is provided through respective sub-commands. To learn more
  about their functionality call the corresponding sub-command with the --help
  flag to see a detailed description.

Options:
  ...

Commands:
  ...
```
For instance, to train a model for classifying neuronal morphologies, we can
check the help of the corresponding sub-command.
```shell
$ morphoclass train --help
Usage: morphoclass train [OPTIONS]

  Train a morphology classification model.

Options:
  ...
```

## Python interface

MorphoClass is a pure Python package, and can be used as any other module once
it is installed.

For instance, we can load a PersLay-based model for morphology classification
and check its documentation as follows.
```python
>>> from morphoclass.models.coriander_net import CorianderNet
>>> help(CorianderNet)

Help on class CorianderNet in module morphoclass.models.coriander_net:

class CorianderNet(torch.nn.modules.module.Module)
 |  CorianderNet(n_classes=4, n_features=64, dropout=False)
 |
 |  A PersLay-based neural network for neuron m-type classification.
 |
 |  Parameters
 |  ----------
 ...
```


# Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain
Project, a research center of the École polytechnique fédérale de Lausanne
(EPFL), from the Swiss government's ETH Board of the Swiss Federal Institutes
of Technology.

Copyright © 2022 Blue Brain Project/EPFL
