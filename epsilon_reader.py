import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import matplotlib.pyplot as plt
import numpy as np
import csv

# for plots
colours = ['r', 'g', 'b', 'k']
models = ['SGD', 'AdamW', 'AdamW LrScheduler', 'AdamW ExpScheduler']

# read and store accuracies
file = open("./adversarial/MNIST_accuracies.csv", "r")
data = list(csv.reader(file, delimiter=","))
model_accuracies = []
for model in data:
    temp_list = [float(x) for x in model]
    model_accuracies.append(temp_list)
file.close()

# Visualise training and validation accuracy
fig = plt.figure()
counter = 0
for model_acc in model_accuracies:
    plt.plot(model_acc, '-o', color=colours[counter], label=models[counter])
    counter += 1
plt.legend(loc='best')
plt.xlabel('epochs trained')
plt.ylabel('accuracy in percent')
plt.xticks(range(0, 21))
plt.yticks(np.arange(0, 105, 10))
# plt.title('Accuracy vs Epsilon')
# plt.tight_layout()
plt.show()
# sys.exit()

# for plots
colours = ['r', 'g', 'b', 'k']
models = ['SGD', 'AdamW', 'AdamW LrScheduler', 'AdamW ExpScheduler']
epsilons = [0, .05, .1, .15, .2, .25, .3]

# read and store epsilon accuracies
file = open("./adversarial/MNIST_epsilons.csv", "r")
data = list(csv.reader(file, delimiter=","))
model_accuracies = []
for model in data:
    temp_list = [float(x) for x in model]
    model_accuracies.append(temp_list)
file.close()

# Visualise training and validation accuracy
fig = plt.figure()
counter = 0
for model_acc in model_accuracies:
    plt.plot(model_acc, '-o', color=colours[counter], label=models[counter])
    counter += 1
plt.legend(loc='best')
plt.xlabel('epsilons')
plt.ylabel('accuracy in percent')
plt.xticks(range(len(epsilons)), epsilons)
plt.yticks(np.arange(0, 105, 10))
# plt.title('Accuracy vs Epsilon')
# plt.tight_layout()
plt.show()



# for plots
colours = ['r', 'g', 'b', 'c', 'y', 'k']
models = ['SGD', 'SGD noDropout', 'AdamW', 'AdamW LrScheduler', 'AdamW ExpScheduler', 'AdamW Exp noDropout']
epsilons = [0, .05, .1, .2, .4, .5, .8]

# read and store epsilon accuracies
file = open("./adversarial/epsilon_accuracies_first.csv", "r")
data = list(csv.reader(file, delimiter=","))
model_accuracies = []
for model in data:
    temp_list = [float(x) for x in model]
    model_accuracies.append([i * 100 for i in temp_list])
file.close()

# Visualise training and validation accuracy
fig = plt.figure()
counter = 0
for model_acc in model_accuracies:
    plt.plot(model_acc, '-o', color=colours[counter], label=models[counter])
    counter += 1
plt.legend(loc='best')
plt.xlabel('epsilons')
plt.ylabel('accuracy in percent')
plt.xticks(range(len(epsilons)), epsilons)
plt.yticks(np.arange(0, 105, 10))
plt.title('Accuracy vs Epsilon')
# plt.tight_layout()
plt.show()