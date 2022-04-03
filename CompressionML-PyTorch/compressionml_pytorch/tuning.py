#!/usr/bin/env python

import chunk
import numpy as np
import os
import PIL
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from functools import partial
from torchvision.transforms import transforms
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split


from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import random

# INFO: Setup accordingly
META_PATH = "meta.h5"
INPUT_PATH = "chunks/"


def filter_dataset(meta_path, metric_name="Compression Rate", min_size=0):
    f = h5py.File(meta_path, "r")
    dset = f["Compression-Trace"]
    winners = {}

    chunk_id = -1
    for trace in dset:
        chunk_id += 1
        if trace["Size"] > min_size:
            if trace["Metric Name"].decode() == metric_name:
                chunk_name = trace["Chunk Name"].decode()
                if (
                    chunk_name not in winners
                    or trace["Metric Measurement"] > winners[chunk_name][1]
                ):
                    winners[chunk_name] = (
                        chunk_id,
                        trace["Metric Measurement"],
                        "{}:{}".format(
                            trace["Compressor name"].decode(), trace["Compressor Level"]
                        ),
                        trace["Compressor Level"],
                    )
    return winners


def dataset_info(dset):
    info = {}
    for item in dset:
        key = dset[item][2]

        if key not in info:
            info[key] = 1
        else:
            info[key] += 1
    return info


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class ChunkDataset(Dataset):
    def __init__(self, items, data_path, dim_size, transform=None):
        self.transform = transform
        self.items = items
        self.items_keys = list(items)
        self.data_path = data_path
        self.dim_size = dim_size
        self.labels_text = list()
        self.idx_to_label = dict()
        self.label_to_idx = dict()

        self.test = 0

        ids_all = []
        for item in items:
            ids_all.append(items[item][2])
        self.labels_text = list(set(ids_all))

        i = 0
        for label_text in self.labels_text:
            self.idx_to_label[i] = label_text
            i += 1

        self.label_to_idx = {v: k for k, v in self.idx_to_label.items()}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        chunk_name = self.items_keys[idx]
        item_path = Path("{}/{}".format(self.data_path, chunk_name))
        chunk = np.fromfile(item_path, dtype=np.float32)
        chunk = chunk[: self.dim_size]
        chunk = chunk.reshape(1, chunk.shape[0])
        # Required: RuntimeError: stack expects each tensor to be equal size, but got [1, 4096] at entry 0 and [1, 2] at entry 8
        chunk = np.resize(chunk, (1, self.dim_size))
        chunk = np.nan_to_num(chunk, copy=True)

        if np.isnan(chunk).any():
            self.test += 1
            print("Nan: ", self.test)

        chunk = torch.from_numpy(chunk)

        if self.transform:
            chunk = self.transform(chunk)
        label = self.items[chunk_name][2]

        return chunk, self.label_to_idx[label]


def train_test_dataset(dataset, val_split=0.15, stratify=None):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=val_split,
        stratify=stratify,
        random_state=0,
    )
    train = Subset(dataset, train_idx)
    test = Subset(dataset, val_idx)
    return (train, test)


def export_info(items, metric, path):
    ids_all = []
    for item in items:
        ids_all.append(items[item][2])
    labels_text = list(set(ids_all))

    exported = ""
    for label in labels_text:
        exported += label + "\n"
    with open(path, "w") as f:
        f.write(metric + "\n\n")
        f.write(exported)

def load_data(input_path, dim_size, winners):
    chunk_dataset = ChunkDataset(winners, input_path, dim_size)
    return train_test_dataset(chunk_dataset)


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes, l_features_in=4096, l_features=512):
        super(NeuralNetwork, self).__init__()
        self.l_features_in = l_features_in
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1 * l_features_in, l_features),
            nn.ReLU(),
            nn.Linear(l_features, l_features),
            nn.ReLU(),
            nn.Linear(l_features, num_classes),
        )

    def forward(self, x):
        # TODO: Might fix issues
        # x = torch.nan_to_num(x)
        x = F.normalize(x)
        x = self.flatten(x)

        pad = nn.ZeroPad2d((0, self.l_features_in - x.size()[1], 0, 0))
        x = pad(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_loss = 0.0
    epoch_steps = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_steps += 1
        if batch % 500 == 499:
            current = batch * len(X)
            # print(f"loss: {running_loss/epoch_steps:>7f}  [{current:>5d}/{size:>5d}]")
            running_loss = 0.0


def val_loop(dataloader, net, criterion):
    # Validation loss
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0

    for i, data in enumerate(dataloader):
        with torch.no_grad():
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_steps += 1

    return (val_loss / len(dataloader), val_steps, total, correct)


def train(config, checkpoint_dir=None, num_classes=None, winners=None):
    net = NeuralNetwork(num_classes, 4096, config["l_features"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    dataset_all = ChunkDataset(winners, INPUT_PATH, 4096)

    # Get all targets
    targets = []
    for _, target in dataset_all:
        targets.append(target)
    targets = torch.tensor(targets)

    # Compute samples weight
    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)]
    )

    weight_all = 1.0 / class_sample_count.float()
    weight_all = weight_all.to(device)

    train_subset, val_subset = train_test_dataset(dataset_all, val_split=0.2)

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        num_workers=4,
        shuffle=True,
    )

    valloader = DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=4
    )

    optimizer = torch.optim.SGD(
        net.parameters(), lr=config["lr"], momentum=config["momentum"]
    )
    criterion = nn.CrossEntropyLoss(weight=weight_all)

    for epoch in range(50):
        train_loop(trainloader, net, criterion, optimizer)
        val_loss, val_steps, total, correct = val_loop(valloader, net, criterion)

        tune.report(loss=val_loss, accuracy=correct / total)

    print("Finished Training")


def search(
    num_samples=10,
    max_num_epochs=10,
    gpus_per_trial=2,
    name="",
    num_classes=None,
    winners=None,
):
    config = {
        # "l_features_in": tune.choice([32, 64, 128, 256, 512, 1024, 2048, 4096]),
        "l_features": tune.choice([64, 128, 256, 512, 1024, 2048]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.0, 0.9),
        "batch_size": tune.choice([16, 32, 64, 128]),
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=3,
    )
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"]
    )
    result = tune.run(
        tune.with_parameters(train, num_classes=num_classes, winners=winners),
        resources_per_trial={"cpu": 6, "gpu": gpus_per_trial},
        config=config,
        name=name,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print(
        "Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]
        )
    )

# for metric in ["Compression Rate", "Compression Speed", "Decompression Speed", "Compression Rate per Time"]:
for metric in ["Compression Rate"]:
    print(metric)
    winners = filter_dataset(META_PATH, metric, min_size=8)
    dataset_item_count = len(winners)
    info = dataset_info(winners)
    print(info)
    num_classes = len(info)

    search(
        num_samples=60,
        max_num_epochs=50,
        gpus_per_trial=1,
        name=metric,
        num_classes=num_classes,
        winners=winners,
    )
