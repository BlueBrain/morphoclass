# Copyright © 2022-2022 Blue Brain Project/EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import device
from torch import nn
from torch.optim import SGD
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset

from morphoclass.training.trainers import Trainer


class TestTrainer:
    @pytest.fixture
    def dataset(self):
        class MyDataset(Dataset):
            def __init__(self):
                super().__init__()
                # One node feature with length one per graph
                self.items = [
                    Data(torch.tensor([[float(i)]]), y=i % 2) for i in range(10)
                ]

            def len(self):
                return len(self.items)

            def get(self, idx):
                return self.items[idx]

        return MyDataset()

    @pytest.fixture
    def net(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_extractor = nn.Linear(1, 3)
                self.classifier = nn.Linear(3, 2)

            def forward(self, batch):
                x = self.feature_extractor(batch.x)
                return self.classifier(x)

        return Net()

    def test_init(self, net, dataset):
        net = net.to("cpu")
        optimizer = ...
        trainer = Trainer(net, dataset, optimizer, DataLoader)
        assert trainer.device == device("cpu")

    def test_loader(self, net, dataset):
        all_features = torch.cat([data.x for data in dataset.items])
        optimizer = ...
        trainer = Trainer(net, dataset, optimizer, DataLoader)

        # No index => load all items in order
        loader = trainer.data_loader()
        batches = [batch.x for batch in loader]
        assert torch.equal(torch.cat(batches), all_features)

        # Can't construct a loader from an empty index set
        with pytest.raises(ValueError, match=r"empty index"):
            trainer.data_loader([])

        # Test batching
        batches_want = torch.tensor([[[0.0], [2.0]], [[5.0], [7.0]]])
        loader = trainer.data_loader([0, 2, 5, 7], batch_size=2)
        assert isinstance(loader, DataLoader)
        batches = [batch.x for batch in loader]
        assert torch.equal(torch.stack(batches), batches_want)

        # Test random batching
        loader = trainer.data_loader([0, 2, 5, 7], batch_size=2, shuffle=True)
        assert isinstance(loader, DataLoader)

        # To test shuffling run the data loader 10 times - it's really unlikely
        # the shuffling would produce the initial order in all cases
        tries = []
        for _ in range(10):
            batches = [batch.x for batch in loader]
            tries.append(torch.stack(batches))
            assert sorted(torch.cat(batches).squeeze().tolist()) == [0.0, 2.0, 5.0, 7.0]
        assert not all(torch.equal(b, batches_want) for b in tries)

    @pytest.mark.parametrize(
        ("labels", "probas", "acc"),
        (
            ([0, 1], [[0.1, 0.9], [0.9, 0.1]], 0.0),
            ([0, 1], [[0.9, 0.1], [0.1, 0.9]], 1.0),
            ([0, 1], [[0.9, 0.1], [0.9, 0.1]], 0.5),
        ),
    )
    def test_acc(self, labels, probas, acc):
        assert Trainer.acc(torch.tensor(labels), torch.tensor(probas)) == acc

    def test_get_latent_features(self, net, dataset):
        optimizer = ...
        trainer = Trainer(net, dataset, optimizer, DataLoader)

        full_loader = trainer.data_loader(batch_size=len(dataset))
        full_batch = next(iter(full_loader))
        features_want = net.feature_extractor(full_batch.x)
        atol = 1e-5

        features = trainer.get_latent_features()
        assert torch.allclose(features_want, features, atol=atol)

        features = trainer.get_latent_features(batch_size=2)
        torch.allclose(features_want, features, atol=atol)

    def test_predict(self, net, dataset):
        optimizer = ...
        trainer = Trainer(net, dataset, optimizer, DataLoader)
        full_loader = trainer.data_loader(batch_size=len(dataset))
        full_batch = next(iter(full_loader))
        out_want = net(full_batch)
        atol = 1e-5

        losses, probas, labels = trainer.predict()
        assert len(losses) == len(dataset)
        assert torch.allclose(out_want, probas, atol=atol)
        assert torch.equal(labels, full_batch.y)

    def test_train(self, net, dataset):
        optimizer = SGD(net.parameters(), lr=1e-3)
        trainer = Trainer(net, dataset, optimizer, DataLoader)
        n_epochs = 5
        batch_size = 2
        train_idx = torch.arange(8)
        val_idx = torch.arange(8, 10)

        # Without val_idx
        history = trainer.train(n_epochs, batch_size, train_idx)
        assert sorted(history) == [
            "train_acc",
            "train_acc_final",
            "train_loss",
            "train_loss_final",
        ]
        assert len(history["train_acc"]) == len(history["train_loss"]) == n_epochs
        assert all(0 <= acc <= 1 for acc in history["train_acc"])
        assert sorted(history["train_loss"], reverse=True) == history["train_loss"]

        # With val_idx
        history = trainer.train(n_epochs, batch_size, train_idx, val_idx=val_idx)
        assert sorted(history) == [
            "predictions",
            "probabilities",
            "train_acc",
            "train_acc_final",
            "train_loss",
            "train_loss_final",
            "val_acc",
            "val_acc_final",
            "val_loss",
            "val_loss_final",
        ]
        assert len(history["train_acc"]) == len(history["train_loss"]) == n_epochs
        assert all(0 <= acc <= 1 for acc in history["train_acc"])
        assert sorted(history["train_loss"], reverse=True) == history["train_loss"]
        assert history["train_acc_final"] == history["train_acc"][-1]
        assert history["train_loss_final"] == history["train_loss"][-1]

        assert len(history["val_acc"]) == len(history["val_loss"]) == n_epochs
        assert all(0 <= acc <= 1 for acc in history["val_acc"])
        assert history["val_acc_final"] == history["val_acc"][-1]
        assert history["val_loss_final"] == history["val_loss"][-1]

        losses, probabilities, labels = trainer.predict(val_idx)
        assert np.allclose(probabilities.numpy(), history["probabilities"], atol=1e-5)
        assert np.array_equal(
            probabilities.numpy().argmax(axis=1), history["predictions"]
        )

        # With load_best
        with pytest.raises(ValueError, match="val_idx"):
            # must provide val_idx
            trainer.train(n_epochs, batch_size, train_idx, load_best=True)
        # hard to test for correct behaviour, at least test nothing breaks
        trainer.train(n_epochs, batch_size, train_idx, val_idx=val_idx, load_best=True)
