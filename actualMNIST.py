# Code is based on https://nextjournal.com/gkoehler/pytorch-mnist (1)
# Through time it has been very modified, but the groundwork was laid by (1)

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
# import time
#
# # Ignore, for comparing speed
# start_time = time.time()

# For faster computation if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting Hyperparameters
n_epochs = 3
batch_size_train = 256
batch_size_test = 1000
learning_rate = 0.01
# for use with SGD optimizer
momentum = 0.5


# Implementing/Initializing DataLoaders
# Note : "The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean
# and standard deviation of the MNIST dataset, we'll take them as a given here."
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

# Defining and building the Network, a simple CNN with two convolutional layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # torch.nn layers are trainable
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            # nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            # nn.Dropout(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(-1, 320)
        x = self.linear_layer(x)
        return F.log_softmax(x, dim=1)


# # All 6 models + optimizers to be tested and attacked
# models_list =  ['model1_SGD', 'model2_SGDnoDrop', 'model3_ADAMW',
#                 'model4_ADAMWlrschedule', 'model5_ADAMWexpschedule',
#                 'model6_ADAMWexpnoDrop']
path_to = 'model2_SGDnoDrop'
path_optimizer = f'{path_to}_optimizer'

# Initializing Network and Optimizer (Stochastic Gradient Descent)
network = Net()
network = network.to(device)
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
# use learning rate scheduler
# scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.3, total_iters=20)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)

# # LOADING PREVIOUS MODELS
# network_state_dict = torch.load(f'./results/{path_to}.pth')
# network.load_state_dict(network_state_dict)
# optimizer_state_dict = torch.load(f'./results/{path_optimizer}.pth')
# optimizer.load_state_dict(optimizer_state_dict)


# TRAINING THE MODEL
#-----------------------------------
# Defining train function
def train(model, curr_optimizer, use_device, curr_loader, loss_list, loss_counter, acc_list, epoch):
    model.train()
    correct = 0
    for batch_id, (data, target) in enumerate(curr_loader):
        data, target = data.to(use_device), target.to(use_device)
        # always set gradients to zero, because PyTorch accumulates them per default
        curr_optimizer.zero_grad()
        output = model(data)
        # negative log likelihood as loss function, because it's ideal for Image Classification
        loss = F.nll_loss(output, target)
        # backpropagation delivers gradients to modify weights and biases based on loss
        loss.backward()
        # update parameters
        curr_optimizer.step()

        # prediction for accuracy purposes
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).sum()
        loss_list.append(loss.item())
        loss_counter.append((batch_id * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
        if batch_id % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(curr_loader.dataset),
                       100. * batch_id / len(curr_loader), loss.item()))
            # Save network and optimizer state for easier reuse and training later
            # torch.save(network.state_dict(), './results/model.pth')
            # torch.save(optimizer.state_dict(), './results/optimizer.pth')

    accuracy = 100. * correct / len(curr_loader.dataset)
    accuracy = accuracy.cpu()
    tester = accuracy.item()
    acc_list.append(accuracy.item())

# Defining test function
def test(model, use_device, curr_loader, loss_list, acc_list):
    model.eval()
    test_loss = 0
    correct = 0
    # accelerates computation (but disables backward() function)
    with torch.no_grad():
        for data, target in curr_loader:
            data, target = data.to(use_device), target.to(use_device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            prediction = output.data.max(1, keepdim=True)[1]
            correct += prediction.eq(target.data.view_as(prediction)).sum()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracy = test_accuracy.cpu()
    loss_list.append(test_loss)
    acc_list.append(test_accuracy.item())
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(curr_loader.dataset),
        100. * correct / len(curr_loader.dataset)))


# TRACKERS to use later for analysis and printouts
train_losses = []
train_accuracies = []
# for plotting accuracies later
train_accuracies.append(np.nan)
#------------------------------
train_counter = []
test_losses = []
test_accuracies = []
# Counter for test epochs for plotting
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


# TRAINING LOOP
#-----------------------------------
# initialize test with randomly initalized parameters
test(network, device, test_loader, test_losses, test_accuracies)
for epoch in range(1, n_epochs + 1):
    train(network, optimizer, device, train_loader, train_losses, train_counter, train_accuracies, epoch)
    # reduce learning rate each epoch
    scheduler.step()
    test(network, device, test_loader, test_losses, test_accuracies)

# # Ignore, for comparing speed
# end_time = time.time()
# print(f'Final time : {end_time - start_time}')
# print(f'\nFinal learning rate : {optimizer.param_groups[0]["lr"]}')
# print(f'\nBest accuracy : {max(test_accuracies)}%')

# PLOTS
#-----------------------------------
# Visualise training and validation loss
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.title('Plot of training and test loss for 20 epochs')
plt.show()

# Visualise training and validation accuracy
fig = plt.figure()
plt.plot(train_accuracies, '-o', color='blue')
plt.plot(test_accuracies, '-o', color='red')
plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy in percent')
#specify axis tick step sizes
plt.xticks(np.arange(0, len(train_accuracies), 1))
plt.yticks(np.arange(0, 105, 10))
plt.title('Plot of training and test accuracy for 20 epochs')
plt.show()