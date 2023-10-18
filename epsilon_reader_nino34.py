import matplotlib.pyplot as plt
import numpy as np
import csv


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