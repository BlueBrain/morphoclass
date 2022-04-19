.. perslay:

PersLay
=======
The persistence images that were used in the :ref:`cnn` section were derived by first
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
in the :ref:`cnn` section::

    diagrams, images, labels = load_persistence_dataset(input_csv_train)

We will only need the diagrams and the labels.

Training
--------
Overall the logic of training the PersLay model is very similar to the one
presented in the :ref:`cnn` section::

    import torch.optim

    import morphoclass as mc
    import morphoclass.models
    import morphoclass.training


    labels_tensor = torch.tensor(labels)
    n_classes = labels.max().item() + 1

    mc.training.reset_seeds(numpy_seed=0, torch_seed=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mc.models.CorianderNet()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    trainer = mc.models.CorianderNetTrainer(model, diagrams, labels, optimizer)
    train_idx = torch.arange(len(labels))
    val_idx = torch.arange(0)
    trainer.train_split(
        train_idx,
        val_idx,
        n_epochs=30000,
        progress_bar_fn=tqdm,
    )


Note that the number of epochs that is necessary to fit this model seems to be higher
than that for other models by a factor of about 100, which was determined empirically.

Evaluation
----------
Also the evaluation is similar to that in the :ref:`CNN` section::

    import numpy as np
    import torch
    from torch.utils.data import DataLoader


    # Data preparation
    diagram_tensors = [
        torch.tensor(diagram / trainer.scale, dtype=torch.float)
        for diagram in diagrams
    ]
    loader = DataLoader(
        diagram_tensors,
        shuffle=False,
        batch_size=1,
        collate_fn=mc.models.CorianderNetTrainer.perslay_collate_diagrams,
    )

    # Evaluation
    model.eval()
    logits = []
    with torch.no_grad():
        for diagram_batch, point_index in loader:
            diagram_batch = diagram_batch.to(device)
            point_index = point_index.to(device)
            batch_logits = model(diagram_batch, point_index).cpu().numpy()
            logits.append(batch_logits)
    if len(logits) > 0:
        logits = np.concatenate(logits)
    else:
        logits = np.array(logits)


    # Compute predictions and accuracy
    predictions = logits.argmax(axis=1)
    acc_train = np.mean(predictions == labels)
    print(f"Accuracy: {acc_train * 100:.2f}%")

Some small differences include:

- There is a ``scale`` variable that is determined by the trainer at training time and is used
  to normalize the values that are used to represent the persistence diagrams. When
  constructing the evaluation set this scale should be used.
- It is necessary to provide a custom collate function in the data loader, since unlike
  for equally-sized images there is no obvious way how several persistence diagrams can
  be collated together to a batch of diagrams. This is the same collate function that
  is used internally by the trainer at training time.
