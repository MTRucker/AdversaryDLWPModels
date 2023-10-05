# Code is based on https://nextjournal.com/gkoehler/pytorch-mnist (1)
# Through time it has been very modified, but the groundwork was laid by (1)
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv

# For faster computation if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting Hyperparameters
batch_size_test = 1000
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
            #nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            #nn.Dropout(),
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
path_to = 'model6_ADAMWexpnoDrop'
path_optimizer = f'{path_to}_optimizer'

# Initializing Network and Optimizer (Stochastic Gradient Descent)
network = Net()
network = network.to(device)
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
# use learning rate scheduler
# scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.3, total_iters=20)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)

# LOADING PREVIOUS MODELS
network_state_dict = torch.load(f'./results/{path_to}.pth')
network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load(f'./results/{path_optimizer}.pth')
optimizer.load_state_dict(optimizer_state_dict)


# Defining test function and adversarial attack within
def test(model, use_device, curr_loader, eps):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in curr_loader:
        data, target = data.to(use_device), target.to(use_device)
        # set requires_grad attribute of tensor. Important for attack, because gradients required
        data.requires_grad = True
        output = model(data)

        # ADVERSARIAL ATTACK HERE
        model.zero_grad()
        loss = F.nll_loss(output, target)
        loss.backward()
        data = data + (eps * data.grad.data.sign())
        #------------------------

        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).sum()

    test_accuracy = 100. * correct / len(curr_loader.dataset)
    test_accuracy = test_accuracy.cpu()
    print("Epsilon: {}\tTest Accuracy = {} / {} = {:.2f}%".format(
        eps, correct, len(curr_loader) * batch_size_test, test_accuracy))
    return test_accuracy.item()


# TRACKERS to use later for analysis and printouts
test_accuracies = []

# RUNNING THE ATTACK
# Run test for each epsilon
for eps in epsilons:
    acc = test(network, device, test_loader, eps)
    test_accuracies.append(acc)

# # write epsilon-accuracies to csv file to later plot a nice overlapping plot
# with open('./adversarial/epsilon_accuracies.csv', 'a', encoding='UTF-8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(test_accuracies)


# Visualise training and validation accuracy
fig = plt.figure()
plt.plot(test_accuracies, '-o', color='blue')
plt.legend(['Test Accuracy'], loc='lower right')
plt.xlabel('epsilons')
plt.ylabel('accuracy in percent')
plt.xticks(range(len(test_accuracies)), epsilons)
plt.yticks(np.arange(0, 105, 10))
plt.title('Accuracy vs Epsilon')
# saveplot with last measured accuracy
# plt.savefig(fname="./adversarial/actual_epsilons/{}-{:.0f}.png".format(
#     path_to, test_accuracies[len(epsilons)-1]))
plt.show()