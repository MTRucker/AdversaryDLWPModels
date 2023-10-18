import math

import matplotlib.pyplot as plt
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
from torch.utils.data import Dataset, DataLoader
from einops.layers.torch import Reduce
from preproc import Normalizer
from collections import Counter

# For faster computation if CUDA is available
device = 'cuda' if th.cuda.is_available() else 'cpu'

# for saving plots and models
tau_to_use = 3

# current model being used (either with or without dropout layer)
current_adam_model = "nodrop_360month"




# ----------------------------------------------------------------------
# CODE FOR DATASET

# preprocessing, generating dataset to be used for DataLoaders
class ElNinoData(Dataset):
    def __init__(self, file, var_label='ts', lat_range: tuple = (-90, 90), lon_range: tuple = (-180, 180),
                 tau: int = 1):
        '''
        file: path to netcdf file
        var_label: variable label in netcdf file
        lat_range: latitude range for data
        lon_range: longitude range for data
        tau: number of time steps to predict
        '''
        # open dataset
        self.ds = xr.open_dataset(file)

        # create lat lon slices
        latitudes = slice(lat_range[0], lat_range[1])
        longitudes = slice(lon_range[0], lon_range[1])

        # nino34 dataset cutout
        self.ds = self.ds.sel(lat=latitudes, lon=longitudes)[var_label]


        # data preprocessing
        # calculates (anomalies and) nino 3.4 index of nino 3.4 cutout
        self.anom, self.nino34_index = self._compute_anomalies_nino34(self.ds)


        # classifies nino 3.4 index/anomalies as either El Nino (2), Neutral (1) or La Nina (0)
        self.nino_label_list = self._label_data(self.ds, self.nino34_index)


        # normalize dataset
        self.normalizer = Normalizer(method='zscore')
        self.ds = self.normalizer.fit_transform(self.ds)
        self.ds.attrs = {'normalizer': self.normalizer}


        # create data tensor
        self.tau = tau
        self.data = th.tensor(self.ds.data, dtype=th.float32)
        self.anom_data = th.tensor(self.anom.data, dtype=th.float32)
        # add 1 dim to signify 1 channel, grayscale (for using Conv2d in NN)
        self.data = th.unsqueeze(self.data, dim=0)


    # function that calculates mean and anomalies of given darray
    def _compute_anomalies_nino34(self, darray):

        # list of all computed nino34 years and months
        labels_list = []

        # first and last year of dataset
        time_start_year = darray['time'].data.min().year
        time_end_year = darray['time'].data.max().year
        time_step_size = 1

        # iterate over all years
        for x in range(time_start_year, time_end_year + (time_step_size * 2), time_step_size):

            # Code to fill the string to the 0001-1200 year format
            time_start = str(x)
            start_len = len(time_start)
            time_end = str(x + 30)
            end_len = len(time_end)

            if start_len < 4:
                while (start_len < 4):
                    time_start = "0" + time_start
                    start_len = len(time_start)

            if end_len < 4:
                while (end_len < 4):
                    time_end = "0" + time_end
                    end_len = len(time_end)

            # edge case so it doesn't do the last 30 years in smaller increments
            if int(time_end) == 1201:
                time_start = time_start + "-01-15"
                time_end = "1201-01-05"
            else:
                time_start = time_start + "-01-15"
                time_end = time_end + "-01-05"

            timeslice_30y = darray.sel(time=slice(time_start, time_end))

            # Calculate mean of Nino 3.4 Index and save within the new dataset
            climatology = timeslice_30y.groupby('time.month').mean(dim='time')
            anom = timeslice_30y.groupby('time.month') - climatology
            nino34 = anom.mean(dim=['lat', 'lon'])
            nino34 = nino34.rolling(time=5, center=True, min_periods=1).mean()

            # edge case so it doesn't do the last 30 years in smaller increments
            if time_end == "1201-01-05":
                for i in range(0, len(nino34.data)):
                    labels_list.append(nino34.data[i])
                anom_datarray = xr.concat([anom_datarray, anom], dim="time")
                break

            sliced_data = nino34.data[0:12]
            for i in range(0, len(sliced_data)):
                labels_list.append(sliced_data[i])

            sliced_anom = anom.isel(time=list(range(0, 12)))
            if time_start == "0001-01-15":
                anom_datarray = anom.isel(time=list(range(0, 12)))
            else:
                anom_datarray = xr.concat([anom_datarray, sliced_anom], dim="time")


        return anom_datarray, labels_list


    # function that categorizes given anomalies to Nino 3.4 Index standard
    def _label_data(self, darray, anomalies):

        elnino_class_counter = 0
        neutral_class_counter = 0
        lanina_class_counter = 0

        # list of all labeled nino34 events within time span (1200 years)
        labels_list = []

        # categorize anomalies as El Nino (2), Neutral (1) or La Nina (0)
        for i in anomalies:
            if i > 0.5:
                labels_list.append(2)
                elnino_class_counter += 1
            elif i < -0.5:
                labels_list.append(0)
                lanina_class_counter += 1
            else:
                labels_list.append(1)
                neutral_class_counter += 1
        print('All Labeled Data Events within Dataset :')
        print(f'La Nina Events : [{lanina_class_counter}/14400]{100. * lanina_class_counter / 14400}')
        print(f'Neutral Events : [{neutral_class_counter}/14400]{100. * neutral_class_counter / 14400}')
        print(f'El Nino Events : [{elnino_class_counter}/14400]{100. * elnino_class_counter / 14400}\n')
        return labels_list


    def __len__(self):
        return self.data.shape[1] - 1


    def __getitem__(self, idx):
            if idx + self.tau > (self.data.shape[1] - 1):
                return self.data[:, idx], self.nino_label_list[(self.data.shape[1] - 1)], self.anom_data[idx]
            return self.data[:, idx], self.nino_label_list[idx + self.tau], self.anom_data[idx]



# INPUTS FOR DATASET: FILEPATH, VAR_LABEL OF FILE, LAT OF CUTOUT, LON OF CUTOUT, MONTHS TO PREDICT#
start_time = time.time()
nino34_dataset = ElNinoData("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc", 'ts', (-5, 5), (-170, -120), tau_to_use)
end_time = time.time()
print(f'Time it took to prepare dataset : {end_time - start_time}')


# Implementing/Initializing DataLoaders (split dataset into 80% training, 20% testing)
n_training = int(len(nino34_dataset) * 0.8)
(train_data, test_data) = th.utils.data.random_split(nino34_dataset, [n_training, len(nino34_dataset) - n_training])
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# THE CODE BELOW IS COURTESY OF JANNIK, THE SUPERVISING(?) STUDENT
# ----------------------------------------------------------------------
# CODE FOR THE ACTUAL NEURAL NETWORK/MODEL

# Defining and building the Network, a simple CNN with two convolutional layers
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



class Nino_classifier(nn.Module):
    '''
    Implementation of a ConvNeXt classifier for SST data.
    '''

    def __init__(self,
                 input_dim: int = 1,
                 latent_dim: int = 128,
                 num_classes: int = 3,
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

# SETTING UP NEURAL NETWORK
# Setting Hyperparameters
batch_size_test = 1
learning_rate = 0.01
# list of epsilon (small) values to use for adversarial perturbations
epsilons = [0, .05, .1, .15, .2, .25, .3]


# Initializing Dataloaders for testing and training
test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)


# Initializing Network and Optimizer (ADAMW)
network = Nino_classifier()
network = network.to(device)
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)


# LOADING PREVIOUS MODELS
network_state_dict = th.load(f'./results/tau_{tau_to_use}_adamw/{current_adam_model}_nino34_model.pth')
network.load_state_dict(network_state_dict)
optimizer_state_dict = th.load(f'./results/tau_{tau_to_use}_adamw/{current_adam_model}_nino34_optimizer.pth')
optimizer.load_state_dict(optimizer_state_dict)
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# DEFINING TEST FUNCTION
def test(model, use_device, curr_loader, eps):
    '''
    model: NN/Model to be used for testing
    use_device: Device on which all tensors should be used
    curr_loader: DataLoader Object used for testing
    '''
    model.eval()
    correct = 0
    adv_examples = []

    for data, target, anom in curr_loader:
        data, target = data.to(use_device), target.to(use_device)
        # set requires_grad attribute of tensor. Important for attack, because gradients required
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
        perturbed_image = data + (eps * data.grad.data.sign())
        perturbation = data - perturbed_image
        # ------------------------

        output = model(perturbed_image)
        final_pred = output.max(1, keepdim=True)[1]

        # check if attack unsuccessful (else save some examples)
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples per tutorial
            if (eps == 0) and (len(adv_examples) < 1):
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

    # calculating loss and accuracy of test
    test_accuracy = 100. * correct / len(curr_loader.dataset)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {:.2f}%".format(
        eps, correct, len(curr_loader) * batch_size_test, test_accuracy))
    return test_accuracy, adv_examples


# THIS IS ONLY FOR SHOWING ONE IMAGE WITH SCALING EPSILONS
def onetest(model, use_device, curr_loader, eps):
    '''
    model: NN/Model to be used for testing
    use_device: Device on which all tensors should be used
    curr_loader: DataLoader Object used for testing
    '''
    model.eval()
    adv_examples = []
    anom_examples = []
    confidences = []
    # avg_confidence = []
    # smallest_conf = 100.0
    # highest_conf = 0.0
    # big_conf_counter = 0

    for data, target, anom in curr_loader:
        data, target, anom = data.to(use_device), target.to(use_device), anom.to(use_device)
        # set requires_grad attribute of tensor. Important for attack, because gradients required
        data.requires_grad = True
        output = model(data)

        # Get Model Confidence
        init_probs = F.softmax(output, dim=-1)
        init_conf = [i for i in init_probs.tolist()[0]]
        init_conf = max(init_conf)
        init_conf = round(init_conf * 100, 2)
        # if init_conf < 95.0:
        #     print(init_conf)
        # if init_conf < smallest_conf:
        #     smallest_conf = init_conf
        # if init_conf > highest_conf:
        #     highest_conf = init_conf
        # if init_conf > 95.0:
        #     big_conf_counter += 1
        # avg_confidence.append(init_conf)
        # continue

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
            perturbed_image = data + (eps[1] * data.grad.data.sign())
            perturbation = data - perturbed_image
            # ------------------------

            output = model(perturbed_image)
            final_pred = output.max(1, keepdim=True)[1]

            # check if attack successful with 0.05 epsilon
            if final_pred.item() != target.item():

                for i in range(0, len(eps)):
                    output = model(data)

                    init_probs = F.softmax(output, dim=-1)
                    init_conf = [i for i in init_probs.tolist()[0]]
                    init_conf = max(init_conf)
                    init_conf = round(init_conf * 100, 2)

                    # get model's prediction
                    initial_pred = output.max(1, keepdim=True)[1]

                    # ADVERSARIAL ATTACK HERE
                    model.zero_grad()
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    perturbed_image = data + (eps[i] * data.grad.data.sign())
                    perturbed_anom = anom + (eps[i] * data.grad.data.sign())
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

                    confidences.append((init_conf, pert_conf, final_conf, pert_pred.item()))

                    adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                    pert = perturbation.squeeze().detach().cpu().numpy()
                    corr_ex = data.squeeze().detach().cpu().numpy()
                    temp_list = [(initial_pred.item(), final_pred.item(), corr_ex, pert, adv_ex)]
                    adv_examples.append(temp_list)
                    corr_anom = anom.squeeze().detach().cpu().numpy()
                    adv_anom = perturbed_anom.squeeze().detach().cpu().numpy()
                    temp_list = [(corr_anom, adv_anom)]
                    anom_examples.append(temp_list)

                break

    # print(f"Average Confidence of Model : {sum(avg_confidence)/len(avg_confidence)}%")
    # print(f"Lowest Conf : {smallest_conf}%")
    # print(f"Highst Conf : {highest_conf}%")
    # print(f"{big_conf_counter}/{len(curr_loader)}, {big_conf_counter/len(curr_loader)}%")
    # sys.exit()
    # calculating loss and accuracy of test
    return adv_examples, anom_examples, confidences


def calibration_function(model, use_device, curr_loader):
    '''
    model: NN/Model to be used for testing
    use_device: Device on which all tensors should be used
    curr_loader: DataLoader Object used for testing
    '''
    model.eval()

    conf_la_counter = [0 for i in range(1, 101)]
    conf_neut_counter = [0 for i in range(1, 101)]
    conf_el_counter = [0 for i in range(1, 101)]

    conf_la_corr = [0 for i in range(1, 101)]
    conf_neut_corr = [0 for i in range(1, 101)]
    conf_el_corr = [0 for i in range(1, 101)]

    # accelerates computation (but disables backward() function)
    with th.no_grad():
        for data, target, anom in curr_loader:
            data, target = data.to(use_device), target.to(use_device)
            output = model(data)

            probs = F.softmax(output, dim=-1)
            conf = [i for i in probs.tolist()[0]]
            conf = max(conf)
            conf = round(conf * 100) - 1

            if target.item() == 0:
                conf_la_counter[conf] += 1
            elif target.item() == 1:
                conf_neut_counter[conf] += 1
            elif target.item() == 2:
                conf_el_counter[conf] += 1

            # get model's prediction
            initial_pred = output.max(1, keepdim=True)[1]

            if initial_pred.item() == target.item():
                if initial_pred.item() == 0:
                    conf_la_corr[conf] += 1
                elif initial_pred.item() == 1:
                    conf_neut_corr[conf] += 1
                elif initial_pred.item() == 2:
                    conf_el_corr[conf] += 1

    correct = [conf_la_corr[i] + conf_neut_corr[i] + conf_el_corr[i] for i in range(len(conf_la_corr))]
    correct_counter = [conf_la_counter[i] + conf_neut_counter[i] + conf_el_counter[i] for i in range(len(conf_la_counter))]

    for i in range(len(correct)):

        if conf_la_counter[i] == 0:
            conf_la_corr[i] = 0
        else:
            conf_la_corr[i] = (conf_la_corr[i] / conf_la_counter[i])*100

        if conf_neut_counter[i] == 0:
            conf_neut_corr[i] = 0
        else:
            conf_neut_corr[i] = (conf_neut_corr[i] / conf_neut_counter[i])*100

        if conf_el_counter[i] == 0:
            conf_el_corr[i] = 0
        else:
            conf_el_corr[i] = (conf_el_corr[i] / conf_el_counter[i])*100

        if correct_counter[i] == 0:
            correct[i] = 0
        else:
            correct[i] = (correct[i] / correct_counter[i])*100

    return conf_la_corr, conf_neut_corr, conf_el_corr, correct



# This is only for plotting correct and incorrect example
def corr_incorr_example(model, use_device, curr_loader):
    '''
    model: NN/Model to be used for testing
    use_device: Device on which all tensors should be used
    curr_loader: DataLoader Object used for testing
    '''
    model.eval()

    # for el nino  events
    lock1elnino = 0
    lock2elnino = 0
    false_elnino_img = []
    # for la nina events
    lock1lanina = 0
    lock2lanina = 0
    false_lanina_img = []

    # accelerates computation (but disables backward() function)
    with th.no_grad():
        for data, target, anom in curr_loader:
            data, target = data.to(use_device), target.to(use_device)
            output = model(data)
            prediction = output.data.max(1, keepdim=True)[1]
            # for-loop just for counting classes and their (in)correct prediction by the model
            for i in range(0, len(prediction.data)):
                prediction_class = prediction.data[i].item()
                actual_class = target.data[i]
                if prediction_class == actual_class:
                    if prediction_class == 0:
                        if lock1lanina == 0:
                            correct_lanina_img = data[i].clone().detach().cpu()
                            lock1lanina += 1
                    if prediction_class == 2:
                        if lock1elnino == 0:
                            correct_elnino_img = data[i].clone().detach().cpu()
                            lock1elnino += 1
                elif prediction_class == 2:
                    if lock2elnino == 0:
                        false_elnino_img.append(data[i].clone().detach().cpu())
                        false_elnino_img.append(actual_class)
                        lock2elnino += 1
                elif prediction_class == 0:
                    if lock2lanina == 0:
                        false_lanina_img.append(data[i].clone().detach().cpu())
                        false_lanina_img.append(actual_class)
                        lock2lanina += 1

    return correct_elnino_img, false_elnino_img, correct_lanina_img, false_lanina_img
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# TESTING ADVERSARIAL ATTACK

# TRACKERS to use later for analysis and printouts
test_accuracies = []
examples = []

# RUNNING THE ATTACK
# start_time = time.time()
# for eps in epsilons:
#     acc, ex = test(network, device, test_loader, eps)
#     test_accuracies.append(acc)
#     examples.append(ex)
# end_time = time.time()
# print(f'Time it took to attack model (batches of 1): {end_time - start_time}')

la_confs, neut_confs, el_confs, all_confs = calibration_function(network, device, test_loader)
# first_la_nan = la_confs.index(next(filter(lambda x: x is not None, la_confs)))
# first_el_nan = el_confs.index(next(filter(lambda x: x is not None, el_confs)))
# first_neut_nan = neut_confs.index(next(filter(lambda x: x is not None, neut_confs)))
# first_nan = min([first_la_nan, first_neut_nan, first_el_nan])

fig = plt.figure()
new_la = []
new_el = []
new_neut = []
new_corr = []

for i in range(5, 105, 5):
    new_la.append(np.percentile(la_confs, i))
    new_neut.append(np.percentile(neut_confs, i))
    new_el.append(np.percentile(el_confs, i))
    new_corr.append(np.percentile(all_confs, i))
first_la_nan = new_la.index(next(filter(lambda x: x != 0, new_la)))
first_el_nan = new_el.index(next(filter(lambda x: x != 0, new_el)))
first_neut_nan = new_neut.index(next(filter(lambda x: x != 0, new_neut)))
first_nan = min([first_la_nan, first_neut_nan, first_el_nan])
for i in range(0, first_la_nan):
    new_la[i] = None
for i in range(0, first_neut_nan):
    new_neut[i] = None
for i in range(0, first_el_nan):
    new_el[i] = None
for i in range(0, first_nan):
    new_corr[i] = None
y = [i for i in range(0, 100)]
for i in range(0, first_nan*5):
    y[i] = None
plt.plot(y, y, color='c')
plt.scatter(y, la_confs, marker='o', color='b', label="La Niña", alpha=0.5)
plt.scatter(y, neut_confs, marker='o', color='g', label="Neutral", alpha=0.5)
plt.scatter(y, el_confs, marker='o', color='r', label="El Niño", alpha=0.5)
plt.plot(y, all_confs, color='k', label="Total Correctness", alpha=0.8)
plt.xlabel("confidence in percent")
plt.ylabel("correctness in percent")
plt.legend(loc='lower center')
plt.show()
sys.exit()

# THIS IS ONLY FOR SHOWING ONE IMAGE WITH SCALING EPSILONS
examples, anoms, confs = onetest(network, device, test_loader, epsilons)


# THIS IS ONLY FOR PLOTTING CORRECT AND INCORRECT EXTREME EVENTS
# corr_elnino_img, fls_elnino_img, corr_lanina_img, fls_lanina_img = corr_incorr_example(network, device, test_loader)
# ----------------------------------------------------------------------




# # THIS IS ONLY FOR PLOTTING CORRECT AND INCORRECT EXTREME EVENTS
# elnino_truth_class = 'Nothing'
# if fls_elnino_img[1] == 0:
#     elnino_truth_class = 'La Nina'
# elif fls_elnino_img[1] == 1:
#     elnino_truth_class = 'Neutral'
#
# lanina_truth_class = 'Nothing'
# if fls_lanina_img[1] == 1:
#     lanina_truth_class = 'Neutral'
# elif fls_lanina_img[1] == 2:
#     lanina_truth_class = 'El Nino'
#
#
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#
# fig.suptitle('Visualisation of predictions of El Nino & La Nina Events whilst testing')
# fig1 = axs[0][0].pcolor(corr_elnino_img.squeeze(), cmap='coolwarm', vmin=-2, vmax=2)
# fig.colorbar(fig1, ax=axs[0][0])
# fig2 = axs[1][0].pcolor(fls_elnino_img[0].squeeze(), cmap='coolwarm', vmin=-2, vmax=2)
# fig.colorbar(fig2, ax=axs[1][0])
# axs[0][0].set_title(f'Correct Classification of El Nino Event')
# axs[1][0].set_title(f'False Classification of El Nino Event\nActual Class {fls_elnino_img[1]}, {elnino_truth_class}')
#
# fig1 = axs[0][1].pcolor(corr_lanina_img.squeeze(), cmap='coolwarm', vmin=-2, vmax=2)
# fig.colorbar(fig1, ax=axs[0][1])
# fig2 = axs[1][1].pcolor(fls_lanina_img[0].squeeze(), cmap='coolwarm', vmin=-2, vmax=2)
# fig.colorbar(fig2, ax=axs[1][1])
# axs[0][1].set_title(f'Correct Classification of La Nina Event')
# axs[1][1].set_title(f'False Classification of La Nina Event\nActual Class {fls_lanina_img[1]}, {lanina_truth_class}')
# plt.show()
# sys.exit()




# ----------------------------------------------------------------------
# PLOTS

# write epsilon-accuracies to csv file to later plot a nice overlapping plot
# with open('./adversarial/nino34_epsilon_accuracies.csv', 'a', encoding='UTF-8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(test_accuracies)


# Visualise new validation accuracy
# fig = plt.figure()
# plt.plot(test_accuracies, '-o', color='blue')
# plt.legend(['Test Accuracy'], loc='lower right')
# plt.xlabel('epsilons')
# plt.ylabel('accuracy in percent')
# plt.xticks(range(len(test_accuracies)), epsilons)
# plt.yticks(np.arange(0, 105, 10))
# plt.title('Accuracy vs Epsilon')
# plt.show()

def name_my_event(number: int = 0):
    if number == 0:
        name = "La Niña"
    elif number == 1:
        name = "Neutral"
    elif number == 2:
        name = "El Niño"
    return name

# Plot each example with a ground truth, original predict, perturbation, and predict
cnt = 0
plt.figure()
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, corr_ex, pert, adv_ex = examples[i][j]
        conf1, conf2, conf3, pert_pred = confs[i]
        plt.pcolor(corr_ex, cmap="coolwarm", vmin=-2, vmax=2)
        if cnt <= 1:
            plt.title(f"Original Image, Class : {name_my_event(orig)}")
            plt.xlabel(f"Predicted Class : {name_my_event(orig)}, {conf1}% Confidence")
        plt.colorbar()
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if cnt <= 2:
            plt.title("Perturbation")
        plt.xlabel(f"Predicted Class : {name_my_event(pert_pred)}, {conf2}% Confidence")
        plt.pcolor(pert, cmap="coolwarm")
        cnt += 1
        plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if cnt <= 3:
            plt.title("Perturbed Image")
        plt.xlabel(f"Predicted Class : {name_my_event(adv)}, {conf3}% Confidence")
        plt.pcolor(adv_ex, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar()
# plt.tight_layout()
plt.show()

cnt = 0
plt.figure()
for i in range(len(epsilons)):
    for j in range(len(anoms[i])):
        cnt += 1
        plt.subplot(len(epsilons), len(anoms[0]) + 1, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig, adv, corr_ex, pert, adv_ex = examples[i][j]
        conf1, conf2, conf3, pert_pred = confs[i]
        corr_anom, adv_anom = anoms[i][j]
        if cnt <= 1:
            plt.title("Original Anomaly")
            plt.xlabel(f"Predicted Class : {name_my_event(orig)}, {conf1}% Confidence")
        plt.pcolor(corr_anom, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar()
        cnt += 1
        plt.subplot(len(epsilons), len(anoms[0]) + 1, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if cnt <= 2:
            plt.title("Perturbed Anomaly")
        plt.xlabel(f"Predicted Class : {name_my_event(adv)}, {conf3}% Confidence")
        plt.pcolor(adv_anom, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar()
plt.tight_layout()
plt.show()
