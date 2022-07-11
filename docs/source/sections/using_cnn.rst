CNN
===
The overall idea of training a convolutional neural network (CNN)
is quite similar to the one for the GNN described in section :doc:`using_gnn`.

At the moment there are some differences in the interface that arose for
historical reasons and might be adjusted in the future if necessary

Loading Data
------------

Unlike GNNs, CNNs operate on images rather than graphs. Therefore the
the morphologies need to be transformed into some kind of 2D representation.
Luckily Lida Kanari has developed a topological framework for doing exactly
that - see the TMD package and the corresponding publications for details.

Here we will be using this TMD approach to obtain the so-called persistence
diagrams of the neurites, and then apply Gaussian kernel-density estimation
to generate images out of the diagrams. Here is a sample code for loading
morphology data together with their TMD persistence diagrams and images::

    import numpy as np
    import torch
    from tmd.Topology.analysis import get_persistence_image_data

    from morphoclass.data import MorphologyDataset
    from morphoclass.features.non_graph import get_tmd_diagrams
    from morphoclass.transforms import BranchingOnlyNeurites
    from morphoclass.transforms import Compose
    from morphoclass.transforms import ExtractEdgeIndex
    from morphoclass.transforms import ExtractTMDNeurites


    def load_persistence_dataset(input_csv):
        # Pre-processing transformations
        pre_transform = [
            ExtractTMDNeurites(neurite_type="apical"),
            BranchingOnlyNeurites(),
            ExtractEdgeIndex(),
        ]
        pre_transform = Compose(pre_transform)

        # Load neurites
        dataset = MorphologyDataset.from_csv(
            csv_file=input_csv,
            pre_transform=pre_transform,
        )

        # Attach TMD diagram to each sample
        for data in dataset:
            data.num_nodes = sum(len(tree.p) for tree in data.tmd_neurites)
        neurite_collection = [data.tmd_neurites for data in dataset]
        tmd_diagrams = get_tmd_diagrams(
            neurite_collection, feature="projection"
        )  # or feature="radial_distance"

        # Normalize TMD diagrams
        xmin, ymin = np.stack([d.min(axis=0) for d in tmd_diagrams]).min(axis=0)
        xmax, ymax = np.stack([d.max(axis=0) for d in tmd_diagrams]).max(axis=0)
        xscale = max(abs(xmax), abs(xmin))
        yscale = max(abs(ymax), abs(ymin))
        scale = np.array([[xscale, yscale]])
        for sample, diagram in zip(dataset, tmd_diagrams):
            sample.diagram = torch.tensor(diagram / scale).float()

        # Attach TMD images
        xmin_norm = min(xmin, 0)
        ymin_norm = min(ymin, 0)
        for sample, diagram in zip(dataset, tmd_diagrams):
            image = get_persistence_image_data(
                diagram,
                xlims=(xmin_norm, xmax),
                ylims=(ymin_norm, ymax),
            )
            image = np.rot90(image)[np.newaxis, np.newaxis]  # shape = (batch, c, w, h)
            sample.image = torch.tensor(image.copy()).float()

        return dataset

As described in section :doc:`data`, we first create a `MorphologyDataset` class, and
then we attach to its sample in this dataset the corresponding persistence diagram and
image.

Training
--------
The convolutional model we propose in this package is defined in `mc.models.CNNet`. Here's
an example of how it can be trained on persistence images::

    import numpy as np
    import torch
    from tqdm import tqdm

    from morphoclass.data.morphology_data_loader import MorphologyDataLoader
    from morphoclass.models import CNNet
    from morphoclass.training import reset_seeds
    from morphoclass.training.trainers import Trainer


    dataset = load_persistence_dataset(input_csv_train)

    all_labels = np.array([s.y for s in dataset])
    label_to_y = dataset.label_to_y
    labels_unique_str = sorted(label_to_y, key=lambda label: label_to_y[label])
    n_classes = len(labels_unique_str)

    reset_seeds(numpy_seed=0, torch_seed=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    moodel = CNNet(image_size=100)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    trainer = Trainer(model, dataset, optimizer, MorphologyDataLoader)
    train_idx = torch.arange(len(dataset))
    val_idx = torch.arange(0)
    history = trainer.train(
        n_epochs=100,
        batch_size=2,
        train_idx=train_idx,
        val_idx=val_idx,
        progress_bar=tqdm,
    )

The main difference is that the trainer class accepts a set of train and validation indices.
The logic here is that one can load a set or morphologies that contains both the train and
validation sets and then specify which of the morphologies should be used in training and which
in validation by providing ``train_idx`` and ``val_idx``, which are sequences of indices.

Here we just want to train on the whole set, so we set ``val_idx`` to an empty sequence, and
``train_idx`` to all indices.

Otherwise the code should be straight-forward and self-explanatory. After running it the model
instance is trained and can be used for prediction.

Evaluating
----------
Unlike for GNNs, the evaluation of the CNN has to be done in a more manual way. This
may change in the future. Let's first look at the code and then make some comments after::

    from torch.utils.data import DataLoader, TensorDataset


    # Create a data loader from the images
    tensor_ds = TensorDataset(images_tensor)
    loader = DataLoader(tensor_ds, batch_size=1, shuffle=False)

    # Run the model on the images
    model.eval()
    logits = []
    with torch.no_grad():
        for batch, in iter(loader):
            batch = batch.to(device)
            batch_logits = model(batch).cpu().numpy()
            logits.append(batch_logits)

    # Transform logits into a numpy array
    if len(logits) > 0:
        logits = np.concatenate(logits)
    else:
        logits = np.array(logits)

    # Compute predictions and accuracy
    predictions = logits.argmax(axis=1)
    acc_train = np.mean(predictions == labels)
    print(f"Accuracy: {acc_train * 100:.2f}%")

As you can see, one needs to manually loop through the data by creating a data loader.
As for the GNN, the output of the model are logits, i.e. logarithms of the probabilities
over the classes. These can be transformed to actual predictions by taking the arg-max,
just as before.
