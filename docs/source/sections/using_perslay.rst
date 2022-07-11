PersLay
=======
The persistence images that were used in the :doc:`using_cnn` section were derived by first
computing the so-called persistence diagrams. Since these diagrams can be considered
as the pure topological representation of the apical graph, without the arbitrariness
that is introduced during the transition to images, one may wonder if it is possible
to train a model directly on such persistence diagrams.

The answer is yes, and one of the methods, called "PersLay", for persistence layer,
was introduced in `arXiv:1904.09378 <https://arxiv.org/abs/1904.09378>`__. It is
an algorithm that takes a persistence diagram as an input and produces and embedding
vector for it. Since a persistence diagram is a set of points in two dimensions, PersLay
can be viewed as a mapping from an arbitrary number of pairs of floating point numbers
into a vector space of a fixed dimension. For more technical details of the implementation
of PersLay please refer to the original publication.

The crucial point of PersLay is that it can be parametrized by trainable weights that
can be optimized through backpropagation, which makes it suitable for a machine learning
model. In the ``morphoclass`` package we propose one possible implementation of such a model.
It is implemented in the class ``morphoclass.models.CorianderNet``.

Data Loading
------------
Loading morphology data, and transforming it into persistence images -- and diagrams -- was
covered in the previous sections, and we can use the helper function defined
in the :doc:`using_cnn` section::

    dataset = load_persistence_dataset(input_csv_train)

We will only need the diagrams and the labels.

Training
--------
Training the PersLay model is almost the same of what explained in the :doc:`using_cnn`
section, we just need to replace ``CNNet`` by ``CorianderNet``::

    import numpy as np
    import torch
    from tqdm import tqdm

    from morphoclass.data.morphology_data_loader import MorphologyDataLoader
    from morphoclass.models import CorianderNet
    from morphoclass.training import reset_seeds
    from morphoclass.training.trainers import Trainer


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = torch.tensor([s.y for s in dataset]).to(device)
    label_to_y = dataset.label_to_y
    labels_unique_str = sorted(label_to_y, key=lambda label: label_to_y[label])
    n_classes = len(labels_unique_str)

    reset_seeds(numpy_seed=0, torch_seed=0)

    model = CorianderNet(n_classes=n_classes, n_features=64)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    trainer = Trainer(model, dataset, optimizer, MorphologyDataLoader)
    train_idx = torch.arange(len(dataset))
    val_idx = torch.arange(0)
    history = trainer.train(
        n_epochs=100,
        batch_size=2,
        train_idx=train_idx,
        val_idx=None,
        progress_bar=tqdm,
    )


Evaluation
----------
The evaluation can be done exactly in the same way explained in the :doc:`using_cnn` section.
The very same code can be used, without need for any change.
