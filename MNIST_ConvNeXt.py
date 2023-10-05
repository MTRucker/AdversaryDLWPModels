# Code is based on https://nextjournal.com/gkoehler/pytorch-mnist (1)
# Through time it has been very modified, but the groundwork was laid by (1)

import torch
import torch as th
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
import csv
from einops.layers.torch import Reduce

# For faster computation if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting Hyperparameters
n_epochs = 20
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


# Define ConvNeXtBlock
class ConvNeXtBlock(nn.Module):
    '''
    Implementation of a ConvNeXt block.
    ConvNeXt block is a residual convolutional block with a depthwise spatial convolution and an inverted bottleneck layer.
    It is a modern variant of the ResNet block, which is especially efficient for large receptive fields.
    '''

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 7, expansion_factor: int = 4,
                 activation_fn=nn.GELU) -> None:
        '''
        input_channels: Number of input channels
        output_channels: Number of output channels
        kernel_size: Kernel size of the depthwise convolution
        expansion_factor: Expansion factor of the inverted bottleneck layer
        activation_fn: Activation function to use
        '''
        super().__init__()
        dim = input_channels * expansion_factor  # Dimension of the inverted bottleneck layer
        # The residual block consists of a depthwise convolution, a group normalization, an inverted bottleneck layer
        # and a projection to the output dimension.
        self.residual = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, groups=input_channels, padding='same'),
            # Process spatial information per channel
            nn.GroupNorm(num_groups=1, num_channels=input_channels),
            # Normalize each channel to have mean 0 and std 1, this stabilizes training.
            nn.Conv2d(input_channels, dim, kernel_size=1),  # Expand to higher dim
            activation_fn(),  # Non-linearity
            nn.Conv2d(dim, output_channels, kernel_size=1),  # Project back to lower dim
        )
        # Shortcut connection to downsample residual dimension if needed
        self.shortcut = nn.Conv2d(input_channels, output_channels,
                                  kernel_size=1) if input_channels != output_channels else nn.Identity()  # Identity if same dim, else 1x1 conv to project to same dim

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)



class Net(nn.Module):
    '''
    Implementation of a ConvNeXt classifier for SST data.
    '''

    def __init__(self,
                 input_dim: int = 1,
                 latent_dim: int = 128,
                 num_classes: int = 10,
                 num_layers: int = 4,
                 downsampling: int = -1,
                 expansion_factor: int = 4,
                 kernel_size: int = 7,
                 activation_fn=nn.GELU):
        '''
        input_dim: Number of input channels
        latent_dim: Number of channels in the latent feature map
        num_classes: Number of classes to classify
        num_layers: Number of ConvNeXt blocks
        downsample_input: Whether to downsample the input with a strided convolution or not
        expansion_factor: Expansion factor of the inverted bottleneck layer
        kernel_size: Kernel size of the depthwise convolutions
        activation_fn: Activation function to use
        '''
        super().__init__()
        # First we need to project the input to the latent dimension
        if downsampling > 0:
            # If we want to downsample the input, we use a strided convolution.
            # This reduces the computational cost of the network a lot.
            assert downsampling % 2 == 0, 'Downsampling factor must be even'
            self.input_projection = nn.Conv2d(input_dim, latent_dim, kernel_size=kernel_size, stride=downsampling,
                                              padding=kernel_size // 2)
        else:
            # If we don't want to downsample the input, we use a 1x1 convolution.
            # This is a cheap operation that doesn't change the spatial dimension.
            self.input_projection = nn.Conv2d(input_dim, latent_dim, 1)
        # Then we process the spatial information with a series of Residual blocks defined above.
        self.cnn_blocks = nn.ModuleList(
            [ConvNeXtBlock(latent_dim, latent_dim, kernel_size, expansion_factor, activation_fn) for _ in
             range(num_layers)])  # List of convolutional blocks
        # Finally, we average the latent feature map and perform classification with an inverted bottleneck MLP.
        self.classifier = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            # Global average pooling, I think this is the same as nn.AdaptiveAvgPool2d(1) but more explicit.
            nn.Linear(latent_dim, latent_dim * expansion_factor),  # Linear layer to expand to higher dim
            activation_fn(),  # Non-linearity
            nn.Linear(latent_dim * expansion_factor, num_classes),  # Final classification layer
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x.shape = (batch_size, input_dim, height, width)
        x = self.input_projection(x)  # (batch_size, latent_dim, height // downsampling, width // downsampling)
        for block in self.cnn_blocks:
            x = block(x)
        logits = self.classifier(x)  # (batch_size, num_classes)
        return logits


# # All 6 models + optimizers to be tested and attacked
# models_list =  ['model1_SGD', 'model2_SGDnoDrop', 'model3_ADAMW',
#                 'model4_ADAMWlrschedule', 'model5_ADAMWexpschedule',
#                 'model6_ADAMWexpnoDrop']
path_to = 'model1_SGD'
# path_optimizer = f'{path_to}_optimizer'

# Initializing Network and Optimizer (Stochastic Gradient Descent)
network = Net()
network = network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
# use learning rate scheduler
# scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.3, total_iters=20)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)

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
        loss = F.cross_entropy(output, target)
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
            torch.save(network.state_dict(), f'./results/new_mnist/{path_to}_model.pth')
            torch.save(optimizer.state_dict(), f'./results/new_mnist/{path_to}_optimizer.pth')

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
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
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
    # scheduler.step()
    test(network, device, test_loader, test_losses, test_accuracies)

# write test accuracies to csv file to later plot a nice overlapping plot
with open('./adversarial/MNIST_accuracies.csv', 'a', encoding='UTF-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(test_accuracies)

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