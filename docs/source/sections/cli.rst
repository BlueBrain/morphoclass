The CLI
=======

The morphoclass package provides a command line interface (CLI) that allows to
train a deep learning model on morphological data, and to use a trained model for
inference. Moreover it provides some basic explain AI (XAI) functionality that allows
to reason about the models predictions.

The entry point for the CLI is called ``morphoclass`` and is automatically installed
together with the ``morphoclass`` package. Running this command shows the
available options and sub-commands.

The available options are

- ``--version``: print the version of the ``morphoclass`` package that
  ``morphoclass`` is associated to
- ``--help``: print available options and sub-commands

The available sub-commands are:

- ``train``: train a model
- ``predict``: usa a trained model for inference
- ``explain``: use a trained model for reasoning on inference

The following sub-sections give more details on each of the three sub-commands

Train
-----
This command can be used to train a machine learning model. Running ``morphoclass train --help``
gives an overview over the available options. The mandatory parameters are:

- ``-m`` or ``--model``: which model to use, can be one of the following:

  - CNN
  - GNN
  - PersLay
  - XGB

- ``-i`` ``--input-csv``: the CSV file specifying the input morphologies. See
  the :ref:`data` section for the details on the formatting.
- ``-o`` or ``--output-dir``: the output directory for the model checkpoint.

The optional parameters are:

- ``-n`` or ``--checkpoint-name``: the name of the checkpoint file.
- ``-e`` or ``--n-epochs``: the number of training epochs.
- ``-s`` or ``--seed``: training seed, useful for reproducibility.

A possible use of the train command can look as follows:

.. code-block:: bash

    $ morphoclass train -m GNN -i my_data.csv -o out -n my_model -e 30 -s 314

After running this command a model will be trained on the input data and the checkpoint of
the trained model will be saved in ``out/my_model.chk``.

.. hint::

    Apart from the trained model weights the checkpoint contains other useful metadata
    like the timestamp or the information about the environment that the model was
    trained in. For manual inspection use ``torch.load`` to load the checkpoint into
    a python dictionary::

        import torch


        chk = torch.load(checkpoint_path)
        print(chk["metadata"]["platform"])


Predict
-------
Once a model has been trained, it can be used to perform inference on new data. This can
be done with the ``morphoclass predict`` command by specifying a model checkpoint and the input
data. Optionally, the vector embeddings of the morphologies can be computed long wiht the
predicted morphology types (as long as supported by the model). The predictions (and
embeddings) are written to a JSON file in the specified output directory. See
``morphoclass predict --help`` for a listing of allowed options.

The mandatory parameters for ``morphoclass predict`` are the following:

- ``-i`` or ``--input-csv``: the CSV file specifying the input morphologies. See
  the :ref:`data` section for the details on the formatting.
- ``-c`` or ``--checkpoint``: the path to the pre-trained model checkpoint. This should
  point to a checkpoint file produced by ``morphoclass train``.
- ``-o`` or ``--output-dir``: the output directory for the predictions.

The optional parameters are:

- ``-n`` or ``--results-name``: The output file name (without the extension). If not
  specified then it will be auto-generated based on the current time and date. The file
  format of the output file will be JSON.
- ``--embed``: if present then also compute the vector embeddings of the morphologies
  (as long as supported by the model) and write them to the same output file.

A possible invocation of ``morphoclass predict`` can look as follows:

.. code-block:: bash

 morphoclass predict -c out/my_model.chk -i new_data.csv -o out -n my_predictions --embed

This loads the pre-trained models from the checkpoint file in ``out/my_model.chk`` and
the morphologies specified in ``new_data.csv``, computes the predicted classes and
embeddings of the morphologies, and stores them in the file ``out/my_predictions.json``.

Explain
-------
There are various techniques in machine learning to try and reason over the predictions
of a model to try and understand which features and characteristics of the input data
lead to the predictions that the model makes. Currently we provide the implementation
of one such algorithm, called Grad-CAM, which has only been implemented for the GNN models.

The available options and parameters can be viewed by using ``morphoclass explain --help``. The
mandatory options are:

- ``-i`` or ``--input-file``: the input morphology file. (Just a single file, not a
  CSV file!)
- ``-c`` or ``--checkpoint``: the path to the pre-trained GNN model checkpoint.
- ``-o`` or ``--output-dir``: the output directory for the results.

There is also an optional parameter:

- ``-n`` or ``--results-name``: the filename (without extension) for the output files.

A possible invocation of the explain command is the following:

.. code-block:: bash

    morphoclass explain -c out/my_model.chk -i my_neuron.h5 -o out -n my_xai

This will load the previously created model checkpoint from ``out/my_model.chk`` and the
neuron morphology from ``my_neuron.h5``. Then the predicted class for the given morphology
will be computed. At prediction time the GNN models is inspected to determine which branching
nodes of the apical dendrites of the input morphology had the highest importance for the
classification decision that the model has made.

These importances as summarized in two plots which in the above examples would be placed in
``out/my_xai_node_saliency.png`` and ``out/my_xai_node_heatmap.png``, and could look as
follows:

|pic1| |pic2|

.. |pic1| image:: ../static/my_xai_node_saliency.png
   :width: 45%

.. |pic2| image:: ../static/my_xai_node_heatmap.png
   :width: 45%

These images are two different representations of the same data and show that in this
particular case the nodes in the tuft of the apical dendrite seemed have contributed
the most to the model's decision to classify this morphology as type TPC-B.
