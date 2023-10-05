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
tau_to_use = 3
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
nino34_dataset = ElNinoData("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc", 'ts', (-5, 5), (-170, -120), tau_to_use)
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
batch_size_test = 128
learning_rate = 0.01


# Initializing Dataloaders for testing and training
test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)


# Initializing Network and Optimizer (ADAMW)
network = Nino_classifier()
network = network.to(device)
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)

# LOADING PREVIOUS MODELS
network_state_dict = th.load(f'./results/tau_{tau_to_use}_adamw/{current_adam_model}_nino34_model.pth')
network_state_dict = {key.replace("module.", ""): value for key, value in network_state_dict.items()}
network.load_state_dict(network_state_dict)
optimizer_state_dict = th.load(f'./results/tau_{tau_to_use}_adamw/{current_adam_model}_nino34_optimizer.pth')
optimizer.load_state_dict(optimizer_state_dict)
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
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
    true_counters = [0, 0, 0]

    correct_list = []
    predicted_list = []

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
                predicted_list.append(prediction_class)
                actual_class = target.data[i].item()
                correct_list.append(actual_class)
                if prediction_class == actual_class:
                    if prediction_class == 0:
                        lanina_counter += 1
                    elif prediction_class == 1:
                        neutral_counter +=1
                    elif prediction_class == 2:
                        elnino_counter += 1
                if actual_class == 0:
                    true_counters[0] += 1
                elif actual_class == 1:
                    true_counters[1] += 1
                elif actual_class == 2:
                    true_counters[2] += 1

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
    print('\n Test Class Accuracies')
    print('\n La Nina Events : {}/{} ({:.0f}%)'.format(lanina_counter, true_counters[0],
                                                       100. * lanina_counter / true_counters[0]))
    print('\n Neutral Events : {}/{} ({:.0f}%)'.format(neutral_counter, true_counters[1],
                                                       100. * neutral_counter / true_counters[1]))
    print('\n El Nino Events : {}/{} ({:.0f}%)\n\n'.format(elnino_counter, true_counters[2],
                                                       100. * elnino_counter / true_counters[2]))
    return correct_list, predicted_list
# ----------------------------------------------------------------------

# initialize test with randomly initalized parameters
corr_list, pred_list = test(network, device, test_loader)
corr_list = corr_list[0:50]
pred_list = pred_list[0:50]
print(len(pred_list))

fig = plt.figure(figsize=(10, 4))
plt.scatter(range(0, len(corr_list)), corr_list, marker='o', color='b', label="Ground Truth", alpha=0.5)
plt.scatter(range(0, len(corr_list)), pred_list, marker='X', color='r', label="Model Prediction", alpha=0.5)
plt.legend(loc=(0.01, 0.2))
plt.xlabel("months")
plt.ylabel("event labels")
plt.locator_params(axis="y", nbins=3)
plt.show()
sys.exit()