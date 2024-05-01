import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ALPHA = 0.05
GAMMA = 0.1

def crush_linear(x):
    y = 0.2 * x + 0.5
    return torch.clamp(y, min = 0, max = 1)

class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        linear = self.linear(x)
        pred = crush_linear(linear)
        return pred

def mf_violation_loss(model, x, prediction, metric, train_dataset):
    sample_idx = torch.randint(high = len(train_dataset))
    sample = train_dataset[sample_idx]

    total_loss = torch.tensor(0., requires_grad=True)
    for applicant in sample:
        pred_p = model(applicant)
        total_loss += torch.max(0, torch.abs(prediction - pred_p) - METRIC(x, applicant))
    return total_loss

def main():
    torch.manual_seed(1)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    features = np.loadtxt(
        open("data/features.csv", "rb"), delimiter=",", skiprows=1, dtype=np.float32
    )
    labels = np.loadtxt(
        open("data/labels.csv", "rb"), delimiter=",", skiprows=1, dtype=np.float32
    )
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

    torch_regressor = LogisticRegression(len(features_train[0]), 1).to(device)

    train_dataset = TensorDataset(features_train, labels_train)
    test_dataset = TensorDataset(features_test, labels_test)
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    # defining the optimizer
    optimizer = torch.optim.Adam(torch_regressor.parameters(), lr=0.1)
    # defining Cross-Entropy loss
    criterion = nn.BCELoss()

    epochs = 10
    for epoch in range(epochs):
        torch_regressor.train()
        for applicants, labels in tqdm(train_dataloader):
            applicants, labels = applicants.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = torch_regressor(applicants)
            loss = criterion(outputs, labels.unsqueeze(1))
            mf_loss = mf_violation_loss(torch_regressor, outputs, METRIC, train_dataset)
            if mf_loss >= ALPHA * GAMMA:
                mf_loss.backward()
            else:
                loss.backward()
            optimizer.step()
        with torch.no_grad():
            correct = 0
            torch_regressor.eval()
            for applicants, labels in test_dataloader:
                outputs = torch_regressor(applicants)
                predicted = outputs.squeeze().round()
                correct += (predicted == labels).sum()
            accuracy = 100 * (correct.item()) / len(test_dataset)
            print('Epoch: {}. Loss: {}. Accuracy: {}'.format(epoch, loss.item(), accuracy))


if __name__ == "__main__":
    main()
