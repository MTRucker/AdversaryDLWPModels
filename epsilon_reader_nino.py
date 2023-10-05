import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

models = [0,  3, 9, 15]
accs = [88, 77, 52, 48]
plt.plot(accs, '-o', color='r')
counter = 0
for i in accs:
    plt.annotate(f"{i}%", (counter - 0.08, i + 5))
    counter += 1
plt.legend(loc='best')
plt.xlabel('lead times (tau)')
plt.ylabel('accuracy in percent')
plt.xticks(range(len(models)), models)
plt.yticks(np.arange(0, 105, 10))
# plt.suptitle('Epsilons')
# plt.tight_layout()
plt.show()
# sys.exit()

# for plots
colours = ['r', 'g', 'b', 'c']
models = ['tau = 0', 'tau = 3', 'tau = 9', 'tau = 15']
epsilons = [0, .05, .1, .15, .2, .25, .3]

# read and store epsilon accuracies
file = open("./adversarial/nino34_epsilon_accuracies.csv", "r")
data = list(csv.reader(file, delimiter=","))
model_accuracies = []
for model in data:
    temp_list = [float(x) for x in model]
    model_accuracies.append([i for i in temp_list])
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
plt.yticks(np.arange(0, 105, 25))
# plt.suptitle('Epsilons')
# plt.tight_layout()
plt.show()

fig2, axs =plt.subplots(ncols=4, sharey=True, sharex=True)
for i in range(0, len(model_accuracies)):
    axs[i].plot(model_accuracies[i], '-o', color=colours[i], label=models[i])
    if i == 0:
        # axs[i].set(xlabel='epsilons', ylabel='accuracy in percent')
        axs[i].set_xticks(range(len(epsilons)), epsilons)
    axs[i].set_box_aspect(1)
    axs[i].legend(loc='best')
# add a big axes, hide frame
fig2.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("epsilons", labelpad=-170)
plt.ylabel("accuracy in percent")
# fig2.text(0.5, 0.04, 'common X', ha='center')
# fig2.text(0.04, 0.5, 'common Y', va='center', rotation='vertical')
# plt.tight_layout()
plt.show()