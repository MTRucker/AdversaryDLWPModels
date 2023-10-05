import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Ignore, for personal use only, not sure if it even makes a slight difference in computation speed
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# Code is based on https://nextjournal.com/gkoehler/pytorch-mnist

# Setting Hyperparameters
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
# setting momentum for use of optimizer, will be explained later
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
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # torch.nn.functional layers not trainable, only functional
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Initializing Network and Optimizer (Stochastic Gradient Descent)
network = Net()
network = network.to(device)
# use Stochastic Gradient Descent with momentum to speed up training (mostly)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


# TRAINING THE MODEL
#-----------------------------------

# Trackers to use later for analysis and printouts
train_losses = []
train_accuracies = []
train_counter = []
test_losses = []
test_accuracies = []
# Counter for test epochs
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

# Defining train function
def train(epoch):
    network.train()
    # for calculating training accuracy
    correct = 0
    #total = 0
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # always set gradients to zero, because PyTorch accumulates them per default
        optimizer.zero_grad()
        output = network(data)
        # negative log likelihood as loss function, because it's ideal for Image Classification
        loss = F.nll_loss(output, target)
        # backpropagation delivers gradients to modify weights and biases based on loss
        loss.backward()
        # update parameters
        optimizer.step()

        # prediction for accuracy purposes
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).sum()
        #total += data[1].size(0)
        if batch_id % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_loader.dataset),
                       100. * batch_id / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_id * 64) + ((epoch - 1) * len(train_loader.dataset)))

            # Save network and optimizer state for easier reuse and training later
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')

    accuracy = 100. * correct / len(train_loader.dataset)
    tester = accuracy.item()
    train_accuracies.append(accuracy.item())

# Defining test function
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # set requires_grad attribute of tensor. Important for Attack
            #data.requires_grad = True
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            prediction = output.data.max(1, keepdim=True)[1]
            correct += prediction.eq(target.data.view_as(prediction)).sum()
            #total += data[1].size(0)
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy.item())
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

# Actual training loop
# initialize test with randomly initalized parameters
test()
train_accuracies.append(test_accuracies[0])
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

# Visualising training and current predictions
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.title('Plot of training and test loss for 3 epochs')
#fig.show()
plt.show()

# testing for accuracy
fig = plt.figure()
plt.plot(train_accuracies, '-o', color='blue')
plt.plot(test_accuracies, '-o', color='red')
plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy in percent')
#specify axis tick step sizes
plt.xticks(np.arange(0, len(train_accuracies), 1))
plt.yticks(np.arange(0, 105, 10))
plt.title('Plot of training and test accuracy for 3 epochs')
#fig.show()
plt.show()

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data, example_targets = example_data.to(device), example_targets.to(device)
with torch.no_grad():
  output = network(example_data)
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
#fig.show()
fig.suptitle('Prediction of model after 3 epochs')
plt.show()


# Loading and continuing training from previously saved state/model
continued_network = Net()
continued_network = continued_network.to(device)
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)

network_state_dict = torch.load('./results/model.pth')
continued_network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load('./results/optimizer.pth')
continued_optimizer.load_state_dict(optimizer_state_dict)

# Running another training loop (From Epoch 3 to 20)
for i in range(4,21):
    test_counter.append(i*len(train_loader.dataset))
    train(i)
    test()

# Visualise new training
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
#fig.show()
plt.title('Plot of training and test loss for 20 epochs')
plt.show()

# again, accuracy
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
#fig.show()
plt.show()

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data, example_targets = example_data.to(device), example_targets.to(device)
with torch.no_grad():
  output = network(example_data)
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
#fig.show()
fig.suptitle('Prediction of model after 20 epochs')
plt.show()