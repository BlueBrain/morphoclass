from __future__ import annotations

import pytest
import torch

import morphoclass.utils
from morphoclass.model_utils import HierarchicalLabels
from morphoclass.models import BidirectionalNet
from morphoclass.models import HBNet
from morphoclass.models import MultiAdjNet

# from morphoclass.models.multi_adj_net import MultiAdjNet
# from morphoclass.models.bidirectional_net import BidirectionalNet
# from morphoclass.models.hbnet import HBNet


@pytest.fixture(scope="session")
def dataset():
    dataset_path = "tests/data"
    dataset, labels, paths = morphoclass.utils.read_layer_neurons_from_dir(
        dataset_path, 5
    )
    morphoclass.utils.normalize_features(dataset)
    return dataset, labels, paths


def test_multi_adj_net(dataset):
    dataset, _, _ = dataset
    model = MultiAdjNet()

    train_idx = list(range(len(dataset)))
    train_loader = morphoclass.utils.get_loader(dataset, train_idx)
    batch = next(iter(train_loader))

    model.eval()
    model.loss_acc(batch)
    model.accuracy(batch)

    # Test attention
    model = MultiAdjNet(attention=True, save_attention=True)
    out = model(batch)
    assert out.shape == (8, 4)
    assert model.pool.last_a_j is not None
    assert model.pool.last_a_j.shape == (720, 1)

    # Test attention + per_feature + save_attention
    model = MultiAdjNet(attention=True, attention_per_feature=True, save_attention=True)
    out = model(batch)
    assert out.shape == (8, 4)
    assert model.pool.last_a_j is not None
    assert model.pool.last_a_j.shape == (720, 512)


def test_man_net(dataset):
    dataset, _, _ = dataset
    model = MultiAdjNet()

    train_idx = list(range(len(dataset)))
    train_loader = morphoclass.utils.get_loader(dataset, train_idx)
    batch = next(iter(train_loader))

    model.eval()
    model.loss_acc(batch)
    model.accuracy(batch)


def test_bidirectional_net(dataset):
    dataset, _, _ = dataset
    model = BidirectionalNet(num_classes=4, num_nodes_features=1)

    train_idx = list(range(len(dataset)))
    train_loader = morphoclass.utils.get_loader(dataset, train_idx)
    batch = next(iter(train_loader))

    model.eval()
    model.loss_acc(batch)
    model.accuracy(batch)


def test_hbnet(dataset):
    data, labels, paths = dataset
    label_dict = {k: v[3:].split("_") for k, v in labels.items()}
    hl = HierarchicalLabels.from_flat_labels(label_dict)

    def get_oh(y):
        return torch.tensor(hl.flat_to_hierarchical_oh(y)).unsqueeze(0)

    for sample in data:
        sample.y_oh = get_oh(sample.y)

    # Model constructor
    class_mask = hl.get_class_segmentation_mask()
    net = HBNet(1, class_mask)

    # Model _apply
    device = torch.device("cpu")
    net = net.to(device)

    # Prepare batch
    train_idx = list(range(len(data)))
    train_loader = morphoclass.utils.get_loader(data, train_idx)
    batch = next(iter(train_loader))

    # forward pass
    log_softmax = net(batch)
    assert list(log_softmax.size()) == [batch.num_graphs, len(class_mask)]

    # Loss
    loss = net.loss(batch)
    assert isinstance(loss, torch.Tensor)

    # Predict
    pred_probas, pred_labels = net.predict(batch)
    assert pred_probas.size()[0] == batch.num_graphs
    assert pred_labels.size()[0] == batch.num_graphs

    # Hierarchical predictions
    flat_proba = net.predict_probabilities(batch)
    parent_mask = torch.tensor(hl.roots, dtype=torch.uint8)
    with pytest.raises(ValueError):
        wrong_proba = flat_proba.unsqueeze(0)
        gen = net.gen_hierarchical_probabilities(wrong_proba, parent_mask, hl)
        next(gen)

    accuracies = net.hierarchical_accuracies(batch, hl)
    for accuracy in accuracies:
        assert 0.0 <= accuracy <= 1.0

    # Hierarchical precision, recall, F1
    result = net.precision_recall_f1(batch, hl)
    assert len(result) == 3

    result = net.precision_recall_f1(batch, hl, average="macro")
    assert len(result) == 3

    result = net.precision_recall_f1(batch, hl, average=None)
    assert len(result) == 3
    for item in result:
        assert len(item) == 4
