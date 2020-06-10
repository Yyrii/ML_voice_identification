import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from machine_learning.net import Net
from torch.utils.data import DataLoader


class Manager:

    def __init__(self, rep_num=10, tt_distribution=[180, 20], batch_size=5, epochs=3, lr=0.001):
        self.rep_num = rep_num  # as data size is too little, for one test
        self.tt_distribution = tt_distribution  # train / test dataset distribution
        self.batch_size = batch_size
        self.epochs = epochs  # full passes over data, for loss minimalisation
        self.learning_rate = lr


    def calculate_efficiency(self, dataset):
        acc = 0
        for _ in range(self.rep_num):
            train_set, test_set = torch.utils.data.random_split(dataset, self.tt_distribution)

            train_batch = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            test_batch = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)

            net = Net()
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)

            for epoch in range(self.epochs):
                for data in train_batch:  # `data` is a batch of data
                    targets, features = data
                    net.zero_grad()
                    features = features.view(-1, dataset.mfcc_length * 12)
                    output = net(features.float())
                    loss = F.nll_loss(output, targets.long())  # calc and grab the loss value
                    loss.backward()  # apply this loss backwards through the network's parameters
                    optimizer.step()  # attempt to optimize weights to account for loss/gradients

            correct = 0
            total = 0

            with torch.no_grad():
                for data in test_batch:
                    targets, features = data
                    output = net(features.view(-1, dataset.mfcc_length * 12).float())
                    for idx, i in enumerate(output):
                        if torch.argmax(i) == targets[idx]:
                            correct += 1
                        total += 1

            acc += correct / total

        return (acc / 10) * 100  # %
