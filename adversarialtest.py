import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv

# import time

# For faster computation if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

learning_rate = 0.01
momentum = 0.5
# list of epsilon (small) values to use for adversarial perturbations
epsilons = [0, .05, .1, .2, .4, .5, .8]


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


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=1, shuffle=True)

# # All 6 models + optimizers to be tested and attacked
# models_list =  ['model1_SGD', 'model2_SGDnoDrop', 'model3_ADAMW',
#                 'model4_ADAMWlrschedule', 'model5_ADAMWexpschedule',
#                 'model6_ADAMWexpnoDrop']
# optimizer_list = ['model1_SGD_optimizer', 'model2_SGDnoDrop_optimizer',
#                   'model3_ADAMW_optimizer', 'model4_ADAMWlrschedule_optimizer',
#                   'model5_ADAMWexpschedule_optimizer', 'model6_ADAMWexpnoDrop_optimizer']
path_to = 'model2_SGDnoDrop.pth'
path_optimizer = 'model2_SGDnoDrop_optimizer.pth'

# Loading and continuing training from previously saved state/model
network = Net()
network = network.to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
# optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.3, total_iters=20)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)
network_state_dict = torch.load(f'./results/{path_to}')
print(f'./results/{path_to}')
print(f'./results/{path_optimizer}')
network.load_state_dict(network_state_dict)
optimizer_state_dict = torch.load(f'./results/{path_optimizer}')
optimizer.load_state_dict(optimizer_state_dict)

# set network to eval for dropouts
network.eval()


# FGSM attack code from PyTorch's Adversarial Attack Tutorial
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + (epsilon * sign_data_grad)
    # Adding clipping to maintain [0,1] range
    perturbation = perturbed_image - image
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image, perturbation


# test function with inbuilt adversarial attack
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
        loss = F.nll_loss(output, target)
        model.zero_grad()
        # backwards pass is absolutely necessary for this adversarial attack
        loss.backward()
        # collect ``datagrad``
        data_grad = data.grad.data
        # execute FGSM
        perturbed_data, perturbation = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]

        # check if attack unsuccessful (else save some examples)
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples per tutorial
            if (epsilon == 0) and (len(adv_examples) < 1):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                pert = perturbation.squeeze().detach().cpu().numpy()
                corr_ex = data.squeeze().detach().cpu().numpy()
                adv_examples.append((initial_pred.item(), final_pred.item(), corr_ex, pert, adv_ex))
        else:
            # Save some adversarial examples for visualization later
            if len(adv_examples) < 1:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
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


# RUNNING THE ATTACK
accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(network, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

# write epsilon-accuracies to csv file to later plot a nice overlapping plot
with open('./adversarial/epsilon_accuracies.csv', 'a', encoding='UTF-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(accuracies)

# plot for comparing accuracy with rising epsilons
plt.figure(figsize=(10, 5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 0.85, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.savefig(fname=f'./adversarial/{path_to}_epsilons_2.png', format='png')
# plt.show()

# # Plot each example with a ground truth, original predict, perturbation, and predict
# cnt = 0
# plt.figure(figsize=(8, 10))
# for i in range(len(epsilons)):
#     for j in range(len(examples[i])):
#         cnt += 1
#         plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         if j == 0:
#             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
#         orig, adv, corr_ex, pert, adv_ex = examples[i][j]
#         plt.title("Original Image")
#         plt.imshow(corr_ex, cmap="gray")
#         cnt += 1
#         plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         plt.title("Perturbation")
#         plt.imshow(pert, cmap="gray")
#         cnt += 1
#         plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
#         plt.xticks([], [])
#         plt.yticks([], [])
#         plt.title("{} -> {}".format(orig, adv))
#         plt.imshow(adv_ex, cmap="gray")
# plt.tight_layout()
# plt.savefig(fname=f'./adversarial/{path_to}_adversarials.png')
# #plt.show()
