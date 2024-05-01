# Rothblum's and Yona's "Probably Approximately Metric-Fair Learning"
# https://proceedings.mlr.press/v80/yona18a/yona18a.pdf


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import DATA_DIR, METRIC_DIR


ALPHA = 0.05
GAMMA = 0.1

metric = None


def crush_linear(x):
    y = 0.2 * x + 0.5
    return torch.clamp(y, min=0, max=1)


def dist(x_inds, y_inds):
    assert metric is not None
    res = [metric[int(x_ind)][int(y_ind)] for x_ind, y_ind in zip(x_inds, y_inds)]
    return torch.tensor(res)


class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        linear = self.linear(x)
        pred = torch.sigmoid(linear)
        return pred


def mf_violation_loss(model, idx, x, prediction, train_dataset):
    sample_idx = torch.randint(
        high=len(train_dataset), size=(len(x),), requires_grad=False
    )
    sample = train_dataset[sample_idx][0]
    sample_idx = sample[:, 0]
    sample = sample[:, 1:]

    pred_p = model(sample)
    print(prediction, pred_p)
    total_loss = torch.max(
        torch.tensor(0.0), torch.abs(prediction - pred_p) - dist(idx, sample_idx)
    )
    return total_loss


def main():
    torch.manual_seed(1)

    features = np.loadtxt(
        open(f"{DATA_DIR}/features.csv", "rb"),
        delimiter=",",
        skiprows=1,
        dtype=np.float32,
    )
    labels = np.loadtxt(
        open(f"{DATA_DIR}/labels.csv", "rb"),
        delimiter=",",
        skiprows=1,
        dtype=np.float32,
    )

    # Add an index column to the features.
    features = np.insert(features, 0, np.arange(len(features)), axis=1)

    np.random.shuffle(features)
    np.random.shuffle(labels)

    features_train = torch.tensor(
        features[: int(len(features) * 0.75)], requires_grad=True
    )
    features_test = torch.tensor(
        features[int(len(features) * 0.75) :], requires_grad=True
    )

    labels_train = torch.tensor(labels[: int(len(features) * 0.75)], requires_grad=True)
    labels_test = torch.tensor(labels[int(len(features) * 0.75) :], requires_grad=True)

    global metric
    metric = np.load(f"{METRIC_DIR}/metric.npy")

    torch_regressor = LogisticRegression(len(features_train[0]) - 1, 1)

    train_dataset = TensorDataset(features_train, labels_train)
    test_dataset = TensorDataset(features_test, labels_test)

    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    optimizer = torch.optim.Adam(torch_regressor.parameters(), lr=0.1)
    criterion = nn.BCELoss()

    epochs = 10
    for epoch in range(epochs):
        torch_regressor.train()
        for applicants, labels in tqdm(train_dataloader):
            idx = applicants[:, 0]
            applicants = applicants[:, 1:]
            optimizer.zero_grad()
            outputs = torch_regressor(applicants)
            loss = criterion(outputs, labels.unsqueeze(1))
            mf_loss = mf_violation_loss(
                torch_regressor, idx, applicants, outputs, train_dataset
            )
            print(mf_loss.mean())
            if mf_loss.mean() >= ALPHA * GAMMA:
                mf_loss.backward()
            else:
                loss.backward()
            optimizer.step()
        with torch.no_grad():
            correct = 0
            torch_regressor.eval()
            for applicants, labels in test_dataloader:
                applicants = applicants[:, 1:]
                outputs = torch_regressor(applicants)
                predicted = outputs.squeeze().round()
                correct += (predicted == labels).sum()
            accuracy = 100 * (correct.item()) / len(test_dataset)
            print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
