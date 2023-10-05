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

# For faster computation if CUDA is available, CURRENTLY ONLY CPU, BECAUSE ERRORS POP UP
device = 'cuda' if th.cuda.is_available() else 'cpu'
CUDA_LAUNCH_BLOCKING="1"

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
        # create lat lon slices
        latitudes = slice(lat_range[0], lat_range[1])
        longitudes = slice(lon_range[0], lon_range[1])
        # open dataset
        self.ds = xr.open_dataset(file)

        # data preprocessing
        # cuts dataset into specified lon and lat range
        self.ds = self.ds.sel(
            lon=self.ds.lon[(self.ds.lon < min(lon_range)) |
                       (self.ds.lon > max(lon_range))],
            lat=slice(np.min(lat_range), np.max(lat_range))
        )[var_label]
        # self.ds = self.ds.sel(lat=latitudes, lon=longitudes)[var_label]
        # calculates means and the resulting anomalies
        self.anom = self._compute_anomalies_nino34(self.ds)
        # classifies anomalies as either El Nino (1), Neutral (0) or La Nina (-1)
        self.nino_label_list = self._label_data(self.ds, self.anom)
        # normalize dataset
        self.normalizer = Normalizer(method='zscore')
        self.ds = self.normalizer.fit_transform(self.ds)
        self.ds.attrs = {'normalizer': self.normalizer}

        # create data tensor
        self.tau = tau
        self.data = th.tensor(self.ds.data)
        # Uncomment below if shape [1, 10, 41] is necessary, currently [10, 41]
        # IF you uncomment this, then you have to follow the comments in __len__ and __getitem__
        self.data = th.unsqueeze(self.data, dim=0)

    # function that calculates mean and anomalies of given darray
    def _compute_anomalies_nino34(self, darray):

        # list of all computed nino34 years and months
        labels_list = []

        # first and last year of dataset
        time_start_year = darray['time'].data.min().year
        time_end_year = darray['time'].data.max().year
        time_step_size = 1
        # time_counter = 0
        # lock = 0

        # iterate over all years
        for x in range(time_start_year, time_end_year + time_step_size, time_step_size):

            # time_counter += 1
            # if time_counter % 30 != 0 and lock == 0:
            #     yeartime_start = time.time()
            #     lock = 1
            # if time_counter % 30 == 0:
            #     lock = 0

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

            time_start = time_start + "-01-15"
            time_end = time_end + "-12-05"

            timeslice_30y = darray.sel(time=slice(time_start, time_end))

            # Calculate mean of Nino 3.4 Index and save within the new dataset
            # yeartime_start = time.time()
            anom = timeslice_30y - timeslice_30y.mean(dim='time')
            # yeartime_end = time.time()
            # print(f'Time to calculate cut_out - cutout.mean(dim=\"time\") : {yeartime_end - yeartime_start}')
            nino34 = anom.mean(dim=['lat', 'lon'])
            sliced_data = nino34.data[0:12]
            for i in range(0, len(sliced_data)):
                labels_list.append(sliced_data[i])
            # if time_counter % 30 == 0:
            #     yeartime_end = time.time()
            #     print(f'\n30 year span calc time : {yeartime_end - yeartime_start}\n')


        return labels_list

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
            if i < -0.5:
                labels_list.append(0)
                lanina_class_counter += 1
            else:
                labels_list.append(1)
                neutral_class_counter += 1
        print('All Labeled Data Events within Dataset :')
        print(f'La Nina Events : [{lanina_class_counter}/14400]{100. * lanina_class_counter / 14400}')
        print(f'Neutral Events : [{neutral_class_counter}/14400]{100. * neutral_class_counter / 14400}')
        print(f'El Nino Events : [{elnino_class_counter}/14400]{100. * elnino_class_counter / 14400}')

        return labels_list

    def __len__(self):
        # return self.data.shape[0] - 1
        # Uncomment below and comment above lines if th.unsqueeze() has been used
        return self.data.shape[1] - 1

    def __getitem__(self, idx):
        # change typeof to either get a PyTorch Tensor or an Xarray Datarray
        # currently set to tensor for NN purposes
        typeof = 'tensor'
        if typeof == 'tensor':
            # return self.data[idx], self.nino_label_list[idx + self.tau]
            # Uncomment below and comment above lines if th.unsqueeze() has been used
            return self.data[:, idx], self.nino_label_list[idx + self.tau]
        elif typeof == 'datarray':
            return self.ds.isel(time=idx), self.nino_label_list[idx + self.tau]


start_time = time.time()
# INPUTS : FILEPATH, VAR_LABEL OF FILE, LAT OF CUTOUT, LON OF CUTOUT, MONTHS TO PREDICT
nino34_dataset = ElNinoData("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc", 'ts', (-31, 32), (130, -70), 3)
end_time = time.time()
print(f'Time it took to prepare dataset : {end_time - start_time}')
# nino34_dataset = ElNinoData("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc", 'ts', (-5, 5), (-170, -120), 9)
# nino_img, nino_label = nino34_dataset


# ----------------------------------------------------------------------
# THE CODE FROM THIS POINT ON HAS BEEN COPIED FROM MY actualMNIST.py file AND MODIFIED
# ----------------------------------------------------------------------
# SETTING UP NEURAL NETWORK
# Setting Hyperparameters
n_epochs = 5
batch_size_train = 5
batch_size_test = 10
learning_rate = 0.01
# for use with SGD optimizer
momentum = 0.5
# list of epsilon (small) values to use for adversarial perturbations
epsilons = [0, .05, .1, .15, .2, .25, .3]

# Implementing/Initializing DataLoaders (split dataset into 80% training, 20% testing)
n_training = int(len(nino34_dataset) * 0.8)
(train_data, test_data) = th.utils.data.random_split(nino34_dataset, [n_training, len(nino34_dataset) - n_training])

train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)


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


# Initializing Network and Optimizer (AdamW)
network = Nino_classifier()
network = network.to(device)
# optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
# use learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)

# x, y = next(iter(train_loader))
# y_hat = network(x)
# print(y_hat)
# print(y_hat.shape)
# sys.exit()

# x, y = next(iter(train_loader))
# print(f'\n\Random Train Data')
# print(y)
# y_hat = network(x)
# print(f'\n\Random Model Data')
# print(y_hat)
# sys.exit()


# # LOADING PREVIOUS MODELS
# network_state_dict = th.load(f'./results/{path_to}.pth')
# network.load_state_dict(network_state_dict)
# optimizer_state_dict = th.load(f'./results/{path_optimizer}.pth')
# optimizer.load_state_dict(optimizer_state_dict)

# TRAINING THE MODEL
# -----------------------------------
# Defining train function
def train(model, curr_optimizer, use_device, curr_loader, epoch):
    loss_list = []
    loss_counter = []
    acc_list = []
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
        loss_counter.append((batch_id * batch_size_train) + ((epoch - 1) * len(curr_loader.dataset)))
        if batch_id % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(curr_loader.dataset),
                       100. * batch_id / len(curr_loader), loss.item()))
            # Save network and optimizer state for easier reuse and training later
            th.save(network.state_dict(), './results/nino34_model.pth')
            th.save(optimizer.state_dict(), './results/nino34_optimizer.pth')

    accuracy = 100. * correct / len(curr_loader.dataset)
    accuracy = accuracy.cpu()
    acc_list.append(accuracy.item())
    return loss_list, loss_counter, acc_list


# Defining test function
def test(model, use_device, curr_loader):
    loss_list = []
    acc_list = []
    model.eval()
    test_loss = 0
    correct = 0
    lanina_counter = 0
    neutral_counter = 0
    elnino_counter = 0
    # accelerates computation (but disables backward() function)
    with th.no_grad():
        for data, target in curr_loader:
            data, target = data.to(use_device), target.to(use_device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            prediction = output.data.max(1, keepdim=True)[1]
            # for i in range(0, len(prediction.data)):
            #     prediction_class = prediction.data[i].item()
            #     actual_class = target.data[i]
            #     if prediction_class == actual_class:
            #         if prediction_class == 0:
            #             lanina_counter += 1
            #         elif prediction_class == 1:
            #             neutral_counter +=1
            #         elif prediction_class == 2:
            #             elnino_counter += 1
            correct += prediction.eq(target.data.view_as(prediction)).sum()
    test_loss /= len(curr_loader.dataset)
    test_accuracy = 100. * correct / len(curr_loader.dataset)
    test_accuracy = test_accuracy.cpu()
    loss_list.append(test_loss)
    acc_list.append(test_accuracy.item())
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(curr_loader.dataset),
        100. * correct / len(curr_loader.dataset)))
    # print('\n Test Class Accuracies')
    # print('\n La Nina Events : {}/{} ({:.0f}%)'.format(lanina_counter, len(curr_loader.dataset),
    #                                                    100. * lanina_counter / len(curr_loader.dataset)))
    # print('\n Neutral Events : {}/{} ({:.0f}%)'.format(neutral_counter, len(curr_loader.dataset),
    #                                                    100. * neutral_counter / len(curr_loader.dataset)))
    # print('\n El Nino Events : {}/{} ({:.0f}%)'.format(elnino_counter, len(curr_loader.dataset),
    #                                                    100. * elnino_counter / len(curr_loader.dataset)))
    return loss_list, acc_list


# TRACKERS to use later for analysis and printouts
train_losses = []
train_accuracies = []
# for plotting accuracies later
train_accuracies.append(np.nan)
# ------------------------------
train_counter = []
test_losses = []
test_accuracies = []
# Counter for test epochs for plotting
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
avg_train_loss = []
avg_train_loss.append(np.nan)

# TRAINING LOOP
# -----------------------------------
# initialize test with randomly initalized parameters
first_loss_list, first_acc_list = test(network, device, test_loader)
test_losses.extend(first_loss_list)
test_accuracies.extend(first_acc_list)
start_time = time.time()
for curr_epoch in range(1, n_epochs + 1):
    loss_list, loss_counter, acc_list = train(network, optimizer, device, train_loader, curr_epoch)
    avg_train_loss.append(sum(loss_list)/len(loss_list))
    train_losses.extend(loss_list)
    train_counter.extend(loss_counter)
    train_accuracies.extend(acc_list)
    # reduce learning rate each epoch
    scheduler.step()
    loss_list, acc_list = test(network, device, test_loader)
    test_losses.extend(loss_list)
    test_accuracies.extend(acc_list)
end_time = time.time()
print(f'Time it took to train model : {end_time - start_time}')
print(f'Model : 4 ConvNetBlocks, {batch_size_train} Train Batchsize, {n_epochs} Epochs')

# PLOTS
# -----------------------------------
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
plt.show()

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
plt.show()
