import matplotlib.pyplot as plt
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import xarray as xr
import numpy as np
import nc_time_axis
import sys
import time
import random
import csv
import math
from torch.utils.data import Dataset, DataLoader
from einops.layers.torch import Reduce
from preproc import Normalizer
from collections import Counter

# For faster computation if CUDA is available
device = 'cuda' if th.cuda.is_available() else 'cpu'

# Setting Hyperparameters
batch_size_test = 1
learning_rate = 0.01
# for use with SGD optimizer
momentum = 0.5
# list of epsilon (small) values to use for adversarial perturbations
epsilons = [0, .05, .1, .15, .2, .25, .3]


# Implementing/Initializing DataLoaders
# Note : "The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean
# and standard deviation of the MNIST dataset, we'll take them as a given here."
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)



# ----------------------------------------------------------------------
# THE CODE BELOW IS COURTESY OF JANNIK, THE SUPERVISING(?) STUDENT
# ----------------------------------------------------------------------
# CODE FOR THE ACTUAL NEURAL NETWORK/MODEL

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
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# THE CODE FROM THIS POINT ON HAS BEEN COPIED FROM MY adversarial attack files AND MODIFIED
# ----------------------------------------------------------------------

# Model used
path_to = "model4_AdamWExp"

# Initializing Network and Optimizer
network = Net()
# network = nn.DataParallel(network)
network = network.to(device)
# optimizer = optim.SGD(network.parameters(), lr=learning_rate)
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.3, total_iters=20)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)


# LOADING PREVIOUS MODELS
network_state_dict = th.load(f'./results/new_mnist/{path_to}_model.pth')
network_state_dict = {key.replace("module.", ""): value for key, value in network_state_dict.items()}
network.load_state_dict(network_state_dict)
optimizer_state_dict = th.load(f'./results/new_mnist/{path_to}_optimizer.pth')
optimizer.load_state_dict(optimizer_state_dict)
# ----------------------------------------------------------------------

def test_only_one(model, device, test_loader, epsilon):

    smallest_confidence = 100.0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        output = model(data)

        init_probs = F.softmax(output, dim=-1)
        init_conf = [i for i in init_probs.tolist()[0]]
        init_conf = max(init_conf)
        init_conf = round(init_conf*100, 2)
        if init_conf < 95.0:
            print(init_conf)
        if init_conf < smallest_confidence:
            smallest_confidence = init_conf

        if init_conf > 95.0:
            continue
        else:

            # get model's prediction
            initial_pred = output.max(1, keepdim=True)[1]
            # ignore attack if prediction wrong
            if initial_pred.item() != target.item():
                continue

            # ADVERSARIAL ATTACK HERE
            model.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            perturbed_image = data + (epsilon * data.grad.data.sign())
            perturbation = data - perturbed_image
            # ------------------------

            output = model(perturbed_image)
            final_pred = output.max(1, keepdim=True)[1]

            final_probs = F.softmax(output, dim=-1)
            final_conf = [i for i in final_probs.tolist()[0]]
            final_conf = max(final_conf)
            final_conf = round(final_conf * 100, 2)


            output = model(perturbation)
            pert_pred = output.max(1, keepdim=True)[1]

            pert_probs = F.softmax(output, dim=-1)
            pert_conf = [i for i in pert_probs.tolist()[0]]
            pert_conf = max(pert_conf)
            pert_conf = round(pert_conf * 100, 2)

            if initial_pred.item() == final_pred.item():
                continue

            print(f"Prediction succesfully changed, from {initial_pred.item()} ({init_conf}% confidence) to {final_pred.item()}")
            # return temporary
            return data.squeeze().detach().cpu().numpy(),\
                   perturbation.squeeze().detach().cpu().numpy(),\
                   perturbed_image.squeeze().detach().cpu().numpy(),\
                   initial_pred.item(), pert_pred.item(), final_pred.item(),\
                   init_conf, pert_conf, final_conf

orig_img, pert, pert_img, corr, pert_class, pred, conf1, conf2, conf3 = test_only_one(network, device, test_loader, 0.2)
# sys.exit()
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
ax1.imshow(orig_img, cmap="gray")
ax1.set_title("Original Image")
ax1.set(xlabel=f"Predicted Class : {corr}\n{conf1}% confidence")
ax1.tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelbottom=False, labelleft=False)
ax2.imshow(pert, cmap="gray")
ax2.set_title("Perturbation")
ax2.set(xlabel=f"Predicted Class : {pert_class}\n{conf2}% confidence")
ax2.tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelbottom=False, labelleft=False)
ax3.imshow(pert_img, cmap="gray")
ax3.set_title("Perturbed Image")
ax3.set(xlabel=f"Predicted Class : {pred}\n{conf3}% confidence")
ax3.tick_params(axis='both', which='both', bottom=False, top=False, right=False, left=False, labelbottom=False, labelleft=False)
# plt.tight_layout()
plt.show()
sys.exit()


# ----------------------------------------------------------------------
# DEFINING TEST FUNCTION
def test(model, device, test_loader, epsilon):
    # counters for tracking
    correct = 0
    adv_examples = []
    # start = time.time()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        output = model(data)
        # get model's prediction
        initial_pred = output.max(1, keepdim=True)[1]
        # ignore attack if prediction wrong
        if initial_pred.item() != target.item():
            continue

        # ADVERSARIAL ATTACK HERE
        model.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        perturbed_image = data + (epsilon * data.grad.data.sign())
        perturbation = data - perturbed_image
        # ------------------------

        output = model(perturbed_image)
        final_pred = output.max(1, keepdim=True)[1]

        # check if attack unsuccessful (else save some examples)
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples per tutorial
            if (epsilon == 0) and (len(adv_examples) < 1):
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                pert = perturbation.squeeze().detach().cpu().numpy()
                corr_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((initial_pred.item(), final_pred.item(), corr_ex, pert, adv_ex))
        else:
            # Save some adversarial examples for visualization later
            if len(adv_examples) < 1:
                adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                pert = perturbation.squeeze().detach().cpu().numpy()
                corr_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((initial_pred.item(), final_pred.item(), corr_ex, pert, adv_ex))

    # Calculate accuracy for current epsilon
    final_acc = correct / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    # end = time.time()
    # print(f'Elapsed time for attack : {end - start}s')

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# TESTING ADVERSARIAL ATTACK

# TRACKERS to use later for analysis and printouts
test_accuracies = []
examples = []

# RUNNING THE ATTACK
start_time = time.time()
for eps in epsilons:
    acc, ex = test(network, device, test_loader, eps)
    test_accuracies.append(acc)
    examples.append(ex)
end_time = time.time()
print(f'Time it took to attack model (batches of 1): {end_time - start_time}')
test_accuracies = [i * 100 for i in test_accuracies]
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# PLOTS

# write epsilon-accuracies to csv file to later plot a nice overlapping plot
# with open('./adversarial/MNIST_epsilons.csv', 'a', encoding='UTF-8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(test_accuracies)


# Visualise new validation accuracy
fig = plt.figure()
plt.plot(test_accuracies, '-o', color='blue')
plt.legend(['Test Accuracy'], loc='lower right')
plt.xlabel('epsilons')
plt.ylabel('accuracy in percent')
plt.xticks(range(len(test_accuracies)), epsilons)
plt.yticks(np.arange(0, 105, 10))
plt.title('Accuracy vs Epsilon')
plt.show()


# # Plot each example with a ground truth, original predict, perturbation, and predict
cnt = 0
plt.figure(figsize=(8, 10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, corr_ex, pert, adv_ex = examples[i][j]
        plt.title("Original Image")
        plt.imshow(corr_ex, cmap="gray")
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("Perturbation")
        plt.imshow(pert, cmap="gray")
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(adv_ex, cmap="gray")
# plt.tight_layout()
plt.show()