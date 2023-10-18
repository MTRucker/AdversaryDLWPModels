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
from torch.utils.data import Dataset, DataLoader
from einops.layers.torch import Reduce
from preproc import Normalizer
from collections import Counter

# For faster computation if CUDA is available
device = 'cuda' if th.cuda.is_available() else 'cpu'

# for saving plots and models
tau_to_use = 15
time_of_script_runs = 0

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



# INPUTS FOR DATASET: FILEPATH, VAR_LABEL OF FILE, LAT OF CUTOUT, LON OF CUTOUT, MONTHS TO PREDICT
start_time = time.time()
nino34_dataset = ElNinoData("/mnt/qb/goswami/data/cmip6/Amon/piControl/CESM2/ts_Amon_CESM2_piControl_r1i1p1f1.nc", 'ts', (-5, 5), (-170, -120), tau_to_use)
end_time = time.time()
print(f'Time it took to prepare dataset : {end_time - start_time}')


# Implementing/Initializing DataLoaders (split dataset into 80% training, 20% testing)
n_training = int(len(nino34_dataset) * 0.8)
(train_data, test_data) = th.utils.data.random_split(nino34_dataset, [n_training, len(nino34_dataset) - n_training])


# labels in training set, used for weighing crossentropy loss later in training
train_classes = [label for _, label, anom_notimportant in train_data]
train_classes = sorted(Counter(train_classes).items())
train_classes = list(dict(train_classes).values())
train_class_weights = []
for i in train_classes:
    train_class_weights.append(1/i)
train_class_weights = th.FloatTensor(train_class_weights)
train_class_weights = train_class_weights.to(device)
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
# THE CODE FROM THIS POINT ON HAS BEEN COPIED FROM MY actualMNIST.py file AND MODIFIED
# ----------------------------------------------------------------------

# SETTING UP NEURAL NETWORK
# Setting Hyperparameters
n_epochs = 50
batch_size_train = 64
batch_size_test = 128
learning_rate = 0.01


# Initializing Dataloaders for testing and training
train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)


# Initializing Network and Optimizer (ADAMW)
network = Nino_classifier()
network = network.to(device)
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# DEFINING TRAIN AND TEST FUNCTIONS

# Defining train function
def train(model, curr_optimizer, use_device, curr_loader, weights_list, epoch):
    '''
    model: NN/Model to be used for training
    curr_optimizer: Optimizer to be used for training
    use_device: Device on which all tensors should be used
    curr_loader: DataLoader Object used for training
    weights_list: List of class weights within training dataset, for use with cross_entropy
    epoch: Current epoch of training
    '''
    loss_list = []
    acc_list = []
    model.train()
    correct = 0
    # variables to count how many of which class were accurately "guessed"
    lanina_counter = 0
    neutral_counter = 0
    elnino_counter = 0

    for batch_id, (data, target, anom) in enumerate(curr_loader):
        data, target = data.to(use_device), target.to(use_device)
        # always set gradients to zero, because PyTorch accumulates them per default
        curr_optimizer.zero_grad()
        output = model(data)
        # cross entropy as loss function, because it's ideal for Image Classification + we need softmax
        loss = F.cross_entropy(output, target)
        # backpropagation delivers gradients to modify weights and biases based on loss
        loss.backward()
        # update parameters
        curr_optimizer.step()

        # prediction for accuracy purposes
        prediction = output.data.max(1, keepdim=True)[1]
        # for-loop just for counting classes and their (in)correct prediction by the model
        for i in range(0, len(prediction.data)):
            prediction_class = prediction.data[i].item()
            actual_class = target.data[i]
            if prediction_class == actual_class:
                if prediction_class == 0:
                    lanina_counter += 1
                elif prediction_class == 1:
                    neutral_counter += 1
                elif prediction_class == 2:
                    elnino_counter += 1

        correct += prediction.eq(target.data.view_as(prediction)).sum()
        loss_list.append(loss.item())
        if batch_id % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(curr_loader.dataset),
                       100. * batch_id / len(curr_loader), loss.item()))
            # Save network and optimizer state for easier reuse and training later
            th.save(network.state_dict(), f'./models/tau_{tau_to_use}_adamw/{current_adam_model}_nino34_model.pth')
            th.save(optimizer.state_dict(), f'./models/tau_{tau_to_use}_adamw/{current_adam_model}_nino34_optimizer.pth')

    accuracy = 100. * correct / len(curr_loader.dataset)
    accuracy = accuracy.cpu()
    acc_list.append(accuracy.item())
    elnino_train_acc = 100. * elnino_counter / len(curr_loader.dataset)
    lanina_train_acc = 100. * lanina_counter / len(curr_loader.dataset)
    return loss_list, acc_list, lanina_train_acc, elnino_train_acc



# Defining test function
def test(model, use_device, curr_loader):
    '''
    model: NN/Model to be used for testing
    use_device: Device on which all tensors should be used
    curr_loader: DataLoader Object used for testing
    '''
    loss_list = []
    acc_list = []
    model.eval()
    test_loss = 0
    correct = 0
    # variables to count how many of which class were accurately "guessed"
    lanina_counter = 0
    neutral_counter = 0
    elnino_counter = 0

    # accelerates computation (but disables backward() function)
    with th.no_grad():
        for data, target, anom in curr_loader:
            data, target = data.to(use_device), target.to(use_device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            prediction = output.data.max(1, keepdim=True)[1]
            # for-loop just for counting classes and their (in)correct prediction by the model
            for i in range(0, len(prediction.data)):
                prediction_class = prediction.data[i].item()
                actual_class = target.data[i]
                if prediction_class == actual_class:
                    if prediction_class == 0:
                        lanina_counter += 1
                    elif prediction_class == 1:
                        neutral_counter +=1
                    elif prediction_class == 2:
                        elnino_counter += 1
            correct += prediction.eq(target.data.view_as(prediction)).sum()
    # calculating loss and accuracy of test
    test_loss /= len(curr_loader.dataset)
    test_accuracy = 100. * correct / len(curr_loader.dataset)
    test_accuracy = test_accuracy.cpu()
    loss_list.append(test_loss)
    acc_list.append(test_accuracy.item())
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(curr_loader.dataset),
        100. * correct / len(curr_loader.dataset)))
    # print class prediction / all classes
    # print('\n Test Class Accuracies')
    # print('\n La Nina Events : {}/{} ({:.0f}%)'.format(lanina_counter, len(curr_loader.dataset),
    #                                                    100. * lanina_counter / len(curr_loader.dataset)))
    # print('\n Neutral Events : {}/{} ({:.0f}%)'.format(neutral_counter, len(curr_loader.dataset),
    #                                                    100. * neutral_counter / len(curr_loader.dataset)))
    # print('\n El Nino Events : {}/{} ({:.0f}%)\n\n'.format(elnino_counter, len(curr_loader.dataset),
    #                                                    100. * elnino_counter / len(curr_loader.dataset)))
    elnino_test_acc = 100. * elnino_counter / len(curr_loader.dataset)
    lanina_test_acc = 100. * lanina_counter / len(curr_loader.dataset)
    return loss_list, acc_list, lanina_test_acc, elnino_test_acc
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# TRAINING LOOP

# TRACKERS to use later for analysis and printouts
train_accuracies = []
train_accuracies.append(np.nan)
avg_train_loss = []
avg_train_loss.append(np.nan)
elnino_train_accs = []
lanina_train_accs = []
elnino_train_accs.append(np.nan)
lanina_train_accs.append(np.nan)
test_accuracies = []
test_losses = []
elnino_test_accs = []
lanina_test_accs = []


# ACTUAL TRAINING LOOP

# initialize test with randomly initalized parameters
first_loss_list, first_acc_list, first_lanina_acc, first_elnino_acc = test(network, device, test_loader)
test_losses.extend(first_loss_list)
test_accuracies.extend(first_acc_list)
lanina_test_accs.append(first_lanina_acc)
elnino_test_accs.append(first_elnino_acc)
start_time = time.time()

# training loop that goes over each epoch in n_epochs
for curr_epoch in range(1, n_epochs + 1):

    # train model for 1 epoch, get loss and accuracy
    loss_list, acc_list,lanina_train_acc, elnino_train_acc = train(network, optimizer, device, train_loader, train_class_weights, curr_epoch)
    avg_train_loss.append(sum(loss_list)/len(loss_list))
    train_accuracies.extend(acc_list)
    lanina_train_accs.append(lanina_train_acc)
    elnino_train_accs.append(elnino_train_acc)

    # reduce learning rate each epoch
    scheduler.step()

    # test model from training, get loss and accuracy
    loss_list, acc_list, lanina_test_acc, elnino_test_acc = test(network, device, test_loader)
    test_losses.extend(loss_list)
    test_accuracies.extend(acc_list)
    lanina_test_accs.append(lanina_test_acc)
    elnino_test_accs.append(elnino_test_acc)


end_time = time.time()
print(f'Time it took to train model : {end_time - start_time}')
print(f'Model : 4 ConvNetBlocks, {batch_size_train} Train Batchsize, {n_epochs} Epochs')
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# PLOTS

# list neccessary for plotting averages
sorted_list = []
for i in range(0, len(test_losses)):
    sorted_list.append(i)


# Visualise training and validation loss
fig = plt.figure()
plt.plot(sorted_list, avg_train_loss, color='blue')
plt.plot(sorted_list, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of epochs')
plt.ylabel('cross entropy loss')
plt.title(f'Plot of training and test loss for {n_epochs} epochs')
plt.savefig(f"./plots/tau_{tau_to_use}_adamw/loss_plot{time_of_script_runs}.png")


# Visualise training and validation accuracy
fig = plt.figure()
plt.plot(train_accuracies, '-o', color='blue')
plt.plot(test_accuracies, '-o', color='red')
plt.legend(['Train Accuracy', 'Test Accuracy'], loc='lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy in percent')
# specify axis tick step sizes
plt.xticks(np.arange(0, len(train_accuracies), 1))
plt.yticks(np.arange(0, 105, 10))
plt.title(f'Plot of training and test accuracy for {n_epochs} epochs')
plt.savefig(f"./plots/tau_{tau_to_use}_adamw/acc_plot{time_of_script_runs}.png")


# Visualise training accuracy for El Nino & La Nina Events
fig = plt.figure()
plt.plot(lanina_train_accs, '-o', color='blue')
plt.plot(elnino_train_accs, '-o', color='red')
plt.plot(np.add(lanina_train_accs, elnino_train_accs).tolist(), '-o', color='black')
plt.legend(['La Nina Accuracy', 'El Nino Accuracy', 'Combined Accuracy'], loc='lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy in percent')
# specify axis tick step sizes
plt.xticks(np.arange(0, len(train_accuracies), 1))
plt.yticks(np.arange(0, 105, 10))
plt.title(f'Plot of extreme events train accuracy for {n_epochs} epochs')
plt.savefig(f"./plots/tau_{tau_to_use}_adamw/elnino_train_plot{time_of_script_runs}.png")


# Visualise testing accuracy for El Nino & La Nina Events
fig = plt.figure()
plt.plot(lanina_test_accs, '-o', color='blue')
plt.plot(elnino_test_accs, '-o', color='red')
plt.plot(np.add(lanina_test_accs, elnino_test_accs).tolist(), '-o', color='black')
plt.legend(['La Nina Accuracy', 'El Nino Accuracy', 'Combined Accuracy'], loc='lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy in percent')
# specify axis tick step sizes
plt.xticks(np.arange(0, len(train_accuracies), 1))
plt.yticks(np.arange(0, 105, 10))
plt.title(f'Plot of extreme events test accuracy for {n_epochs} epochs')
plt.savefig(f"./plots/tau_{tau_to_use}_adamw/elnino_test_plot{time_of_script_runs}.png")
