import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    body = nn.Sequential(
      nn.Linear(dim, hidden_dim),
      norm(hidden_dim),
      nn.ReLU(),
      nn.Dropout(drop_prob),
      nn.Linear(hidden_dim, dim),
      norm(dim)
    )
    return nn.Sequential(nn.Residual(body), nn.ReLU())

    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
      nn.Linear(dim, hidden_dim),
      nn.ReLU(),
      *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
      nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
      model.train()
    else:
      model.eval()
    loss_fn = nn.SoftmaxLoss()
    loss_sum = []
    acc_sum = []
    num_samples = 0
    for X, y in dataloader:
      
      X = X.reshape((X.shape[0], -1))
      y_hat = model(X)
      loss = loss_fn(y_hat, y)
      if opt:
        opt.reset_grad()
        loss.backward()
        opt.step()
      y_hat_np = y_hat.numpy()
      y_np = y.numpy()
      acc = np.sum(np.argmax(y_hat_np, axis=1)==y_np) 
      acc_sum.append(acc)
      loss_sum.append(loss.data.numpy())
      num_samples += X.shape[0]

    return 1 - np.sum(acc_sum) / num_samples, np.mean(np.array(loss_sum))
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(data_dir+"/train-images-idx3-ubyte.gz",
                            data_dir+"/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(data_dir+"/t10k-images-idx3-ubyte.gz",
                            data_dir+"/t10k-labels-idx1-ubyte.gz")
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = MLPResNet(28*28, hidden_dim)
    opt = optimizer(params=model.parameters(), weight_decay=weight_decay, lr=lr)
    for i in range(epochs):
      train_avg_err, train_avg_loss = epoch(train_dataloader, model, opt=opt)
      print(f"acc: {1.0 - train_avg_err:.4f}, Loss: {train_avg_loss:.4f}")
      
      test_avg_err, test_avg_loss = epoch(test_dataloader, model)
      print(f"acc: {1.0 - test_avg_err:.4f}, Loss: {test_avg_loss:.4f}")
    return train_avg_err, train_avg_loss, test_avg_err, test_avg_loss
    
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
