DVC
===

DVC plays hand-in-hand with the CLI part of MorphoClass. In fact, often CLI
commands were specifically written to be used in DVC. Of course these commands
can still be used as usual in an interactive shell or in scripts.

If you're not familiar with DVC we recommend that you consult the official
`DVC documentation <https://dvc.org/doc>`__ for an introduction. In order
to follow this section we recommend that you're familiar with the following
points:

* Idea of DVC
* Basic DVC commands
* DVC configuration, remotes
* Tracking files using DVC
* DVC pipelines and DAGs
* ``dvc.yaml``, ``dvc.lock``, and ``params.yaml``

The MorphoClass DVC stages defined in ``dvc.yaml`` can be divided into the
following logical groups

* Data pre-processing
* Feature extraction
* Training
* Evaluation
* Reports

The following sections address all of these groups.

Additionally two other kinds of functionality are currently in development:

* XAI
* Transfer learning

Data pre-processing
-------------------

All data related to these stages is located in the directory ``dvc/data/``.

The data processing workflow can be summarised in the following diagram:

.. code-block::

    (1) raw => (2) preprocess => (3) MCAR => (4) organise => (5) final dataset

1. Raw morphologies:

    * external input, tracked with DVC, origin of all pipelines
    * currently 3 datasets (see ``dvc/data/raw/``)

        * pyramidal cells (PC)
        * interneurons (IN)
        * Janelia

2. Preprocess:

    * main goal: produce a CSV file that can be consumed by MCAR
    * additional steps for PC+IN using ``morphoclass preprocess-dataset`` (see
      docstring)

        * Read a given database file in ``MorphDB`` format
        * Read the morphologies from the given directory. The morphology paths
          have to match those in the database file.
        * Remove rows with equal paths in the database. This can happen if the
          same morphology is assigned to two different cortical layers. We
          don't need the information on the cortical layer and therefore can
          discard the duplicates.
        * Remove all m-type classes with only one morphology.
        * Find and report duplicate morphologies using ``morph_tool``. It can
          happen that different morphology files with different file names
          contain the same morphology.
        * For interneurons only all morphologies where the m-type contains "PC"
          or is equal to "L4_SCC" will be dropped.
        * A report with all the actions taken will be saved to disk.

    * Janelia: just reformat the CSV file for MCAR,
      ``dvc/data/preprocess-janelia.py``. (no ``MorphDB`` file)

3. MCAR:

    * essentially just calling
      ``morphology_workflows --local-scheduler Curate``.
    * processing steps: ``Collect -> CheckNeurites -> Sanitize -> Recenter -> Orient -> Align -> Resample.``
    * see ``dvc/data/mcar-luigi.cfg`` for additional info, note stages with
      ``skip = true``.
    * to do: currently a bit wasteful on disk space - all intermediate
      results are tracked; can we do better?

4. Organise:

    * organise morphologies into sub-directories by m-type
    * create ``dataset.csv`` files both for the whole dataset and per layer
    * output in ``dvc/data/final/<dataset name>``

The DAG for the interneuron data stages shown below shows once more how
different stages relate to each other:

.. code-block::

    $ dvc dag organise-dataset@interneurons
                                      +---------------------------+
                                      | data/raw/interneurons.dvc |
                                      +---------------------------+
                                   ******                         ******
                              *****                                     *****
                           ***                                               *****
    +---------------------------------+                                           ***
    | preprocess-dataset@interneurons |                                      *****
    +---------------------------------+                                 *****
                                   ******                         ******
                                         *****               *****
                                              ***         ***
                                      +----------------------------+
                                      | mcar-curation@interneurons |
                                      +----------------------------+
                                                     *
                                                     *
                                                     *
                                    +-------------------------------+
                                    | organise-dataset@interneurons |
                                    +-------------------------------+

Feature extraction
------------------
We implement the feature extraction as a separate stage that precedes the
training. The corresponding CLI command is

.. code-block::

    $ morphoclass extract-features

The rationale behind setting up a separate stage/command for feature
extraction is that once extracted the features are saved to disk and can
be re-used by different training stages. This saves a considerable amount
of time and speeds up the training. Moreover, having the features pre-extracted
and saved to disk allows to inspect them to make sure the feature extraction
works as intended.

The corresponding DVC stages start with the prefix ``features-`` and the
outputs are written to ``dvc/extract-features/``.

The command ``morphoclass extract-features`` takes a CSV file that specifies
a morphology dataset and extracts one of the following features:

* ``graph-rd``: graph features with radial distances
* ``graph-proj``: graph features with distances to the y-axis
  (projection onto the y-axis)
* ``diagram-tmd-rd``: TMD persistence diagram with radial distances as
  filtration function.
* ``diagram-tmd-proj``: TMD persistence diagram with y-axis projection features.
* ``diagram-deepwalk``: persistence diagram with deepwalk features (if deepwalk
  is installed).
* ``image-tmd-rd``: TMD persistence image with radial distances as
  filtration function.
* ``image-tmd-proj``: TMD persistence image with y-axis projection features.
* ``image-deepwalk``: persistence image with deepwalk features (if deepwalk
  is installed).

.. note::

    The ``deepwalk`` feature extractors are not activated by default since
    DeepWalk's licence does not allow us to install it as a dependency. To
    use it please install the package manually. See the :doc:`installation`
    section for instructions.

After running the command, the extracted features are saved to disk, in the
directory specified as a command-line argument. For each morphology a separate
feature file is created.

For additional information and options please see
``morphoclass extract-features --help``.

Training
--------

* CLI command: ``morphoclass train``.
* DVC stages: ``dvc train@...`` and ``dvc train-xxx`` (see ``dvc.yaml``)
* Directory: ``dvc/training/``
* Parametrized through (see ``morphoclass train --help`` for details)

    * ``--features-dir``: the pre-extracted features
    * ``--model-config``: model configuration YAML file
    * ``--splitter-config``: splitter configuration YAML file



Example model config
....................

.. code-block:: yaml

    batch_size: 2
    n_epochs: 100
    model_class: morphoclass.models.CorianderNet
    model_params:
      n_features: 64
    optimizer_class: torch.optim.Adam
    optimizer_params:
      lr: 0.005
      weight_decay: 0.0005

* ``batch_size``: the batch size for deep learning models
* ``model_class``: the model class, should be importable; we use:

    * ``morphoclass.models.CNNet``
    * ``morphoclass.models.ManNet`` (=GNN)
    * ``morphoclass.models.CorianderNet`` (=PersLay)
    * ``xgboost.XGBClassifier``
    * ``sklearn.tree.DecisionTreeClassifier``

* ``model_params``: class-specific parameters, to be used via
  ``model_class(**model_params)``
* ``optimizer_class``: the optimizer class, analogous to ``model_class``,
  only for deep learning
* ``optimizer_params``: analogous to ``model_params``, to be used via
  ``optimizer_class(**optimizer_params)``.

Example splitter config
.......................

.. code-block:: yaml

    splitter_class: sklearn.model_selection.StratifiedKFold
    splitter_params:
      n_splits: 3

* ``splitter_class``: an scikit-learn splitter class, analogous
  to ``model_class``
* ``splitter_params``: analogous to ``model_params``, to be used
  via ``splitter_class(**splitter_params)``

Output of the ``morphoclass train`` command
...........................................

* The parameter ``--checkpoint-dir <out-dir>`` specifies the output directory
  for the checkpoint
* ``<out-dir>/checkpoint.chk``: the checkpoint with the trained model and
  other metadata that serves to completely reproduce the training setup.
* ``<out-dir>/images/``: legacy images, will be removed in the future.

.. admonition:: to do

    * Remove the creation of the ``<out-dir>/images folder``
    * Replace ``--checkpoint-dir`` by ``--checkpoint-path``

Evaluation
----------

The ``morphoclass evaluate`` allows to computed various statistics and
figures on a trained checkpoint produced by the ``morphoclass train`` command.

There are three different sub-sub-commands (see
``morphoclass evaluate --help``):

* ``latent-features``: generate plots of latent features (DL models only)
* ``outliers``: visualize CleanLab outlier morphologies
* ``performance``: generate a model performance report

Outdated stages
---------------
The following legacy sub-commands have been transformed and superseded by other
sub-commands. They should no longer be used.

* ``feature-extractor``: superseded by the ``extract-features`` command
* ``training-and-evaluating``: superseded by ``train`` and ``evaluate``
* ``performance-report``: superseded by ``morphoclass evaluate`` and
  ``morphoclass performance-table``

DVC Cache
---------
This sub-section is a word of caution when using DVC in development together
with a remote.

Every time ``dvc repro`` is run the output is added to the DVC cache, even if
the results have not been recorded by adding ``dvc.lock`` and other output
files to git. A subsequent ``dvc push`` will push all of this to the remote.
This can lead to a lot of unnecessary files in the cache and the remote that
aren't necessary and can't even be accessed.

In this context the ``dvc gc`` command can be quite helpful. It allows to
removed unused data from the DVC cache prior to pushing data to the remote.
It is also possible use this command to prune data directly on the remote.
We refer to the
`official DVC documentation <https://dvc.org/doc/command-reference/gc>`_
for more details.
