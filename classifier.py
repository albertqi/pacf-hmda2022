import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def one_hot(arr, dim):
    """Returns one-hot representation of y."""
    res = np.zeros((len(arr), dim))
    for i in range(len(arr)):
        res[i][arr[i]] = 1
    return res

class SoftmaxRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.W = None
        self.runs = 200

    def fit(self, X, y):
        """
        Fit the weights W of softmax regression using gradient descent with L2 regularization
        in the form (lambda/2) * norm(w)^2
        Use the results from Problem 2 to find an expression for the gradient

        :param X: a 2D numpy array of (transformed) feature values. Shape is (n x 2)
        :param y: a 1D numpy array of target values (Dwarf=0, Giant=1, Supergiant=2).
        :return: None
        """
        # Initializing the weights (do not change!)
        # The number of classes is 1 + (the highest numbered class)
        # augment matrix
        X = np.hstack((np.array([[1] for i in range(len(X))]), np.array(X)))
        num_classes = 1 + y.max()
        num_features = X.shape[1]
        self.W = np.ones((num_classes, num_features)).T

        print(X.shape, self.W.shape, one_hot(y, num_classes).shape)
        for i in tqdm(range(self.runs)):
            lhs = softmax(X @ self.W) - one_hot(y, num_classes)  # k x n
            grad = X.T @ lhs + self.lam * self.W  # k x d
            self.W -= self.eta * grad
        self.W = self.W.T

        print(self.W)

    def predict(self, X_pred):
        """
        The code in this method should be removed and replaced! We included it
        just so that the distribution code is runnable and produces a
        (currently meaningless) visualization.

        Predict classes of points given feature values in X_pred

        :param X_pred: a 2D numpy array of (transformed) feature values. Shape is (n x 2)
        :return: a 1D numpy array of predicted classes (Dwarf=0, Giant=1, Supergiant=2).
                 Shape should be (n,)
        """
        X_pred = np.hstack((np.array([[1] for i in range(len(X_pred))]), np.array(X_pred)))  # add bias
        y_pred = np.array(softmax(X_pred @ self.W.T))
        y_pred = [np.argmax(y) for y in y_pred]
        return np.array(y_pred)

    def predict_proba(self, X_pred):
        """
        Predict classification probabilities of points given feature values in X_pred

        :param X_pred: a 2D numpy array of (transformed) feature values. Shape is (n x 2)
        :return: a 2D numpy array of predicted class probabilities (Dwarf=index 0, Giant=index 1, Supergiant=index 2).
                 Shape should be (n x 3)
        """
        X_pred = np.hstack((np.array([[1] for i in range(len(X_pred))]), X_pred))  # add bias
        y_pred = np.array(softmax(X_pred @ self.W.T))
        return y_pred

class LogisticRegression(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)
    
    def forward(self, x):
        linear = self.linear(x)
        # pred = torch.sigmoid(linear)
        return linear
        

def main():
    features = np.loadtxt(open("data/features.csv", "rb"), delimiter=",", skiprows=1, dtype=np.float32)
    labels = np.loadtxt(open("data/labels.csv", "rb"), delimiter=",", skiprows=1, dtype=np.float32)

    features_train = torch.tensor(features[:int(len(features) * 0.75)], requires_grad=True)
    features_test = torch.tensor(features[int(len(features) * 0.75):], requires_grad=False)

    labels_train = torch.tensor(labels[:int(len(features) * 0.75)], requires_grad=True)
    labels_test = torch.tensor(labels[int(len(features) * 0.75):], requires_grad=False)

    torch_regressor = LogisticRegression(len(features_train[0]), 1)
    
    train_dataset = TensorDataset(features_train, labels_train)
    test_dataset = TensorDataset(features_test, labels_test)
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    # defining the optimizer
    optimizer = torch.optim.Adam(torch_regressor.parameters(), lr=0.001)
    # defining Cross-Entropy loss
    criterion = nn.BCELoss()

    epochs = 50
    for epoch in range(epochs):
        torch_regressor.train()
        for applicants, labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs = torch_regressor(applicants)
            loss = criterion(outputs, labels.unsqueeze(1))
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

    # print(features.shape, features_train.shape)
    # print(labels.shape, labels_train.shape)

    # regressor = SoftmaxRegression(eta=1e-3, lam=1e-3)
    # regressor.fit(features_train, labels_train)

    # predictions = regressor.predict(features_test)
    # correct = np.sum(predictions == labels_test)
    # print(correct / len(labels_test))


if __name__ == "__main__":
    main()
