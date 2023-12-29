GNN
===

In this section we outline how a graph-based neural network  (GNN) can be trained
and evaluated on morphology data. We will be using a high-level interface
that does *not* require to manually set up a training loop nor to explicitly
do the forward and backward passes.

Loading Data
------------
Before training any model one needs to load the data, which is described in the
:doc:`data` section. Here is an example of how this could be done for the
concrete use case at hand.

First we define a helper function for data loading::

    import morphoclass as mc
    import morphoclass.transforms
    import morphoclass.training


    def load_dataset(input_csv, fitted_scaler=None):
        pre_transform = mc.transforms.Compose([
            mc.transforms.ExtractTMDNeurites(neurite_type='apical'),
            mc.transforms.BranchingOnlyNeurites(),
            mc.transforms.ExtractEdgeIndex(),
        ])

        dataset = mc.data.MorphologyDataset.from_csv(
            csv_file=input_csv,
            pre_transform=pre_transform,
        )

        feature_extractor = mc.transforms.ExtractRadialDistances()

        transform, fitted_scaler = mc.training.make_transform(
            dataset=dataset,
            feature_extractor=feature_extractor,
            n_features=1,
            fitted_scaler=fitted_scaler,
        )
        dataset.transform = transform

        return dataset, fitted_scaler

The benefit of a helper function is that it can be used for the training, validation,
and the test sets, whithout having to duplicate code. It loads data specified in the
given CSV file, and extracts the radial distance features from the branching-only
nodes of the apicals.

One caveat of data loading is that the feature scalers should be fitted to the training
data, and then re-used for the validation and test data. This is why this helper function
accepts a second optional argument ``fitted_scaler``. If it is not provided then a new
scaler is fitted to the data, otherwise the one provided will be used.

One can see that internally a new function, ``mc.training.make_transform``, is called. Its
purpose is to combine feature extraction and feature scaling into one transform, more or less
along the same lines that were presented in the :doc:`data` section. For maximal additional
details see the API documentation and the source code.

Given this helper function the data can be loaded as follows::

    dataset_train, fitted_scaler = load_dataset(input_csv_train)
    dataset_val, _ = load_dataset(input_csv_val, fitted_scaler=fitted_scaler)
    dataset_test, _ = load_dataset(input_csv_test, fitted_scaler=fitted_scaler)


Training
--------
Also several GNN models can be found in this package, the currently recommended model
is ``mc.models.ManNet``. It can be trained on the data loaded above in a few lines of code::

    import torch
    import torch.optim
    from tqdm import tqdm

    import morphoclass as mc
    import morphoclass.models
    import morphoclass.training
    import morphoclass.utils


    mc.utils.make_torch_deterministic()
    mc.training.reset_seeds(numpy_seed=0, torch_seed=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mc.models.ManNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    man_net_trainer = mc.models.ManNetTrainer(model, dataset_train, optimizer)
    man_net_trainer.train(n_epochs=300, progress_bar_fn=tqdm)

Most of this code should be self-explanatory, here are a few additional comments:

- The first two lines after the imports are optional, but should be used if reproducible
  training results are desired.
- If no GPU is available, one may set ``device = "cpu"`` directly.
- The ``lr`` parameter in the optimizer is the learning rate. It was chosen empirically
  and the user may experiment with other values.
- The ``n_epochs`` parameter is also an empirical value, and may be changed during experimentation.
- The prograss bar parameter is optional, typical values are ``tqdm.tqdm`` like above
  or ``tqdm.notebook.tqdm`` for jupyter notebooks.

Evaluating
----------
The evaluation of the trained models can be done using the same trainer interface as defined
in the previous subsection using trainers ``predict()`` method. Note that this method returns
logarithms of the probabilities over the different morphology classes, and the predicted
classes need to be computed by hand using ``argmax``. Here's and example how to the accuracy
of the trained model on the training set can be calculated::

    logits_train = mannet_trainer.predict()
    predictions_train = logits_train.argmax(axis=1)
    labels_train = np.array([sample.y for sample in dataset_train])
    acc_train = np.mean(predictions_train == labels_train)

    print(f"Training accuracy: {acc_train * 100:.2f}%")

A few complimentary comments:

- In order to evaluate the model on the validation or the test set a new instance
  of the ``ManNetTrainer`` class needs to be instantiated with the trained model,
  a different dataset, and an arbitrary optimizer (``None`` can also be provided for
  the optimizer).
- The logits returned by the ``trainer.predict()`` method have the shape ``(n_samples, n_classes)``.
- In order to translate the auto-generated numerical class labels to human-readable ones
  one can use the ``dataset.class_dict`` dictionary::

    for prediction in predictions_train:
        print(dataset_train.class_dict[prediction])
