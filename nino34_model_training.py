import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import xarray as xr
import numpy as np
import nc_time_axis
from torch.utils.data import Dataset, DataLoader
from einops.layers.torch import Reduce
from preproc import Normalizer

# For faster computation if CUDA is available
device = 'cuda' if th.cuda.is_available() else 'cpu'

# Determines how many months ahead the current model needs to predict (see line after ElNinoData() function)
# This also affects the path of saving the model during training (see middle of train() function)
tau_to_use = 3



# ----------------------------------------------------------------------
# CODE FOR DATASET

# Preprocessing, generating Dataset to be used for DataLoaders
class ElNinoData(Dataset):
    def __init__(self, file, var_label='ts', lat_range: tuple = (-5, 5), lon_range: tuple = (-170, -120),
                 tau: int = 1):
        '''
        file: path to netcdf file
        var_label: variable label in netcdf file
        lat_range: latitude range for data
        lon_range: longitude range for data
        tau: number of time steps/months to predict
        '''
        # Open Dataset
        self.ds = xr.open_dataset(file)

        # Create lat-lon slices
        latitudes = slice(lat_range[0], lat_range[1])
        longitudes = slice(lon_range[0], lon_range[1])

        # Retrieve specified cutout (per default set to Niño 3.4 [5N-5S, 170W-120W]) from Dataset
        self.ds = self.ds.sel(lat=latitudes, lon=longitudes)[var_label]


        # Data Preprocessing
        # Calculates anomalies and Niño 3.4 Index of Niño 3.4 cutout
        self.anom, self.nino34_index = self._compute_anomalies_nino34(self.ds)

        # Classifies nino 3.4 index/anomalies as either El Nino (2), Neutral (1) or La Nina (0)
        self.nino_label_list = self._label_data(self.ds, self.nino34_index)

        # Normalize cutout (utilizing Jakob's preproc.py script)
        self.normalizer = Normalizer(method='zscore')
        self.ds = self.normalizer.fit_transform(self.ds)
        self.ds.attrs = {'normalizer': self.normalizer}


        # Create data tensor for use with PyTorch
        self.tau = tau
        self.data = th.tensor(self.ds.data, dtype=th.float32)
        self.anom_data = th.tensor(self.anom.data, dtype=th.float32)
        # Add 1 dim to signify 1 channel, grayscale (for using Conv2d in model below)
        self.data = th.unsqueeze(self.data, dim=0)
        # NOTE : anom_data was not squeezed, for it's use is purely for visuals
        #        (and to be used within this function)


    # Function that calculates mean and anomalies of given DataArray
    def _compute_anomalies_nino34(self, darray):

        # This function computes the Niño 3.4 Index values within 30-year windows,
        # which are shifted by 1 year for all years within the given dataset.
        # e.g. 0001-0030, 0002-0031, 0003-0032, ..., 1171-1200

        # List of all computed Niño 3.4 Index values
        nino34_values_list = []

        # First and last year of Dataset
        time_start_year = darray['time'].data.min().year
        time_end_year = darray['time'].data.max().year
        time_step_size = 1

        # Iterate over all years
        for x in range(time_start_year, time_end_year + (time_step_size * 2), time_step_size):

            # Code to fill the string to the 0001-1200 year format required by the netcdf file
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

            # Edge case if so it doesn't do the last 30 years in smaller increments
            if int(time_end) == 1201:
                time_start = time_start + "-01-15"
                time_end = "1201-01-05"
            else:
                time_start = time_start + "-01-15"
                time_end = time_end + "-01-05"


            # Retrieve DataArray slice at specified 30-year window (360 months)
            timeslice_30y = darray.sel(time=slice(time_start, time_end))

            # Calculate mean of Nino 3.4 Index and save within the new dataset
            # Step 1 : Compute area averaged total SST from Niño 3.4 region (already given by monthly format)
            # Step 2 : Compute monthly climatology for area averaged total SST from Niño 3.4 region,
            #          and subtract climatology from area averaged total SST time series to obtain anomalies.
            # Step 3 : Smooth the anomalies with a 5-month running mean.
            climatology = timeslice_30y.groupby('time.month').mean(dim='time')
            anom = timeslice_30y.groupby('time.month') - climatology
            nino34 = anom.mean(dim=['lat', 'lon'])
            nino34 = nino34.rolling(time=5, center=True, min_periods=1).mean()


            # If last 30 years reached, then add all 360 months, not just first 12
            if time_end == "1201-01-05":
                for i in range(0, len(nino34.data)):
                    nino34_values_list.append(nino34.data[i])
                anom_datarray = xr.concat([anom_datarray, anom], dim="time")
                break

            # Retrieve only first year (12 months) of calculated 30-year interval to be added
            # Add the Niño 3.4 Index values to labels_list
            sliced_data = nino34.data[0:12]
            for i in range(0, len(sliced_data)):
                nino34_values_list.append(sliced_data[i])

            # Add the anomalies to anomaly dataarray
            sliced_anom = anom.isel(time=list(range(0, 12)))
            if time_start == "0001-01-15":
                anom_datarray = anom.isel(time=list(range(0, 12)))
            else:
                anom_datarray = xr.concat([anom_datarray, sliced_anom], dim="time")


        return anom_datarray, nino34_values_list


    # Function that categorizes given anomalies to Nino 3.4 Index standard
    def _label_data(self, darray, anomalies):

        # Counters for printouts
        elnino_class_counter = 0
        neutral_class_counter = 0
        lanina_class_counter = 0

        # List of all labeled Niño 3.4 events within Dataset time span (1200 years)
        labels_list = []

        # Categorize anomalies as El Niño (2), Neutral (1) or La Niña (0)
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


        # Print ratio of labeled events
        print('All Labeled Data Events within Dataset :')
        print(f'La Nina Events : [{lanina_class_counter}/14400] {100. * lanina_class_counter / 14400}')
        print(f'Neutral Events : [{neutral_class_counter}/14400] {100. * neutral_class_counter / 14400}')
        print(f'El Nino Events : [{elnino_class_counter}/14400] {100. * elnino_class_counter / 14400}\n')


        return labels_list


    # usual PyTorch len function
    def __len__(self):
        return self.data.shape[1] - 1


    # usual PyTorch getitem function
    def __getitem__(self, idx):

            # If the month to be predicted lies outside of the Dataset, just return the last month of the Dataset.
            # This potentially skews the last few months, as they do not predict "correctly", but required to limit reach of getitem
            if idx + self.tau > (self.data.shape[1] - 1):
                return self.data[:, idx], self.nino_label_list[(self.data.shape[1] - 1)], self.anom_data[idx]


            return self.data[:, idx], self.nino_label_list[idx + self.tau], self.anom_data[idx]



# INPUTS FOR DATASET: FILEPATH, VAR_LABEL OF FILE, LAT OF CUTOUT, LON OF CUTOUT, MONTHS TO PREDICT
nino34_dataset = ElNinoData("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc", 'ts', (-5, 5), (-170, -120), tau_to_use)


# Implementing/Initializing DataLoaders (split dataset into 80% training, 20% testing)
n_training = int(len(nino34_dataset) * 0.8)
(train_data, test_data) = th.utils.data.random_split(nino34_dataset, [n_training, len(nino34_dataset) - n_training])
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# THE CODE (+ COMMENTS) BELOW IS COURTESY OF JANNIK THUEMMEL, THE SUPERVISING STUDENT OF THIS THESIS
# ----------------------------------------------------------------------
# CODE FOR THE ACTUAL NEURAL NETWORK/MODEL
# IF YOU WANT VISUAL CLARIFICATION FOR THE MODEL'S ARCHITECTURE, LOOK ON THE README OF THE GITHUB REPOSITORY

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
# SETTING UP NEURAL NETWORK

# Setting Hyperparameters
n_epochs = 10
batch_size_train = 64
batch_size_test = 128
learning_rate = 0.01


# Initializing Dataloaders for testing and training
train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)


# Initializing Network and Optimizer (ADAMW with exponential learning rate scheduler)
network = Nino_classifier()
network = network.to(device)
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)


# Loading previous Models and Optimizers, if needed
# network_state_dict = th.load(f'./results/tau_{tau_to_use}_adamw/{current_adam_model}_nino34_model.pth')
# This line below is optional, but I needed it, as it allowed me to unpack CUDA saved models. My advice; keep it
# network_state_dict = {key.replace("module.", ""): value for key, value in network_state_dict.items()}
# network.load_state_dict(network_state_dict)
# optimizer_state_dict = th.load(f'./results/tau_{tau_to_use}_adamw/{current_adam_model}_nino34_optimizer.pth')
# optimizer.load_state_dict(optimizer_state_dict)
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# DEFINING TRAIN AND TEST FUNCTIONS

# Defining train function
def train(model, curr_optimizer, use_device, curr_loader, epoch):
    '''
    model: NN/Model to be used for training
    curr_optimizer: Optimizer to be used for training
    use_device: Device on which all tensors should be used
    curr_loader: DataLoader Object used for training
    epoch: Current epoch of training
    '''
    loss_list = []
    acc_list = []
    model.train()
    correct = 0
    # Variables to count how many of which class were accurately "guessed"
    lanina_counter = 0
    neutral_counter = 0
    elnino_counter = 0

    # Iterate over every datapoint in the training DataLoader
    for batch_id, (data, target, anom) in enumerate(curr_loader):

        data, target = data.to(use_device), target.to(use_device)

        # Always set gradients to zero, because PyTorch accumulates them per default
        curr_optimizer.zero_grad()
        output = model(data)
        # Cross entropy as loss function (ideal for image classification, you can also add softmax for clearer prediciton confidence)
        loss = F.cross_entropy(output, target)
        # Backpropagation delivers gradients to modify weights and biases based on loss
        loss.backward()
        # Update parameters
        curr_optimizer.step()

        # Prediction for accuracy purposes
        prediction = output.data.max(1, keepdim=True)[1]


        # For-loop just for counting classes and their (in)correct prediction by the model
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


        # If correct, increase correct
        correct += prediction.eq(target.data.view_as(prediction)).sum()
        # Add loss to loss_list
        loss_list.append(loss.item())


        # This patch of code prints the progress of training and saves the model + optimizer every so often
        if batch_id % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(curr_loader.dataset),
                       100. * batch_id / len(curr_loader), loss.item()))
            # Save network and optimizer state for easier reuse and training later
            th.save(network.state_dict(), f'./results/adamw_tau{tau_to_use}_nino34_model.pth')
            th.save(optimizer.state_dict(), f'./results/adamw_tau{tau_to_use}_nino34_optimizer.pth')


    # Calculate overall accuracy of current training
    # [.cpu() necessary to detach accuracy from CUDA, because of correct]
    accuracy = 100. * correct / len(curr_loader.dataset)
    accuracy = accuracy.cpu()
    acc_list.append(accuracy.item())
    # Calculate extreme events accuracy of current training
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
    # Variables to count how many of which class were accurately "guessed"
    lanina_counter = 0
    neutral_counter = 0
    elnino_counter = 0

    # Accelerates computation (but disables backward() function)
    with th.no_grad():

        # Iterate over every datapoint in the training DataLoader
        for data, target, anom in curr_loader:

            data, target = data.to(use_device), target.to(use_device)
            output = model(data)
            # Cross entropy as loss function
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            # Prediction for accuracy purposes
            prediction = output.data.max(1, keepdim=True)[1]


            # For-loop just for counting classes and their (in)correct prediction by the model
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


            # If correct, increase correct
            correct += prediction.eq(target.data.view_as(prediction)).sum()


    # Calculating loss and accuracy of test
    test_loss /= len(curr_loader.dataset)
    test_accuracy = 100. * correct / len(curr_loader.dataset)
    test_accuracy = test_accuracy.cpu()
    loss_list.append(test_loss)
    acc_list.append(test_accuracy.item())

    # Print loss and accuracy of test
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(curr_loader.dataset),
        100. * correct / len(curr_loader.dataset)))
    # OPTIONAL : Print class prediction / all classes
    # print('\n Test Class Accuracies')
    # print('\n La Nina Events : {}/{} ({:.0f}%)'.format(lanina_counter, len(curr_loader.dataset),
    #                                                    100. * lanina_counter / len(curr_loader.dataset)))
    # print('\n Neutral Events : {}/{} ({:.0f}%)'.format(neutral_counter, len(curr_loader.dataset),
    #                                                    100. * neutral_counter / len(curr_loader.dataset)))
    # print('\n El Nino Events : {}/{} ({:.0f}%)\n\n'.format(elnino_counter, len(curr_loader.dataset),
    #                                                    100. * elnino_counter / len(curr_loader.dataset)))

    # Calculate extreme events accuracy of current training
    elnino_test_acc = 100. * elnino_counter / len(curr_loader.dataset)
    lanina_test_acc = 100. * lanina_counter / len(curr_loader.dataset)


    return loss_list, acc_list, lanina_test_acc, elnino_test_acc
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# TRAINING AND TESTING LOOP

# TRACKERS to use later for analysis and printouts
# NaN's just make plotting easier
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


# Initialize test with randomly initalized parameters
first_loss_list, first_acc_list, first_lanina_acc, first_elnino_acc = test(network, device, test_loader)
test_losses.extend(first_loss_list)
test_accuracies.extend(first_acc_list)
lanina_test_accs.append(first_lanina_acc)
elnino_test_accs.append(first_elnino_acc)


# Training + Testing loop that goes over each epoch in n_epochs
for curr_epoch in range(1, n_epochs + 1):

    # Train model for 1 epoch, get loss and accuracy
    loss_list, acc_list,lanina_train_acc, elnino_train_acc = train(network, optimizer, device, train_loader, curr_epoch)
    avg_train_loss.append(sum(loss_list)/len(loss_list))
    train_accuracies.extend(acc_list)
    lanina_train_accs.append(lanina_train_acc)
    elnino_train_accs.append(elnino_train_acc)

    # Reduce learning rate each epoch for scheduler (optimizer)
    scheduler.step()

    # Test model that was trained, get loss and accuracy
    loss_list, acc_list, lanina_test_acc, elnino_test_acc = test(network, device, test_loader)
    test_losses.extend(loss_list)
    test_accuracies.extend(acc_list)
    lanina_test_accs.append(lanina_test_acc)
    elnino_test_accs.append(elnino_test_acc)


# Printout just for easier understanding
print(f'Model : 4 ConvNetBlocks, {batch_size_train} Train Batchsize, {n_epochs} Epochs')
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# PLOTS

# List neccessary for plotting averages
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


# ----------------------------------------------------------------------
# THE PLOTS BELOW ARE WRONG, AS THEY CALCULATE EXTREME EVENT ACCURACY BASED ON ALL DATA, NOT JUST THEIR RESPECTIVE EVENTS.
# THIS CAN BE REPAIRED BY HAVING SPECIAL COUNTERS FOR ONLY THE EXTREME EVENTS WITHIN THE TRAIN AND TEST FUNCTIONS.

# THESE PLOTS ARE NOT REALLY IMPORTANT NOR PARTICULARLY USEFUL, WHICH IS WHY THEY ARE NOT PROPERLY FIXED.
# THE CODE BELOW WILL RUN WITHOUT PROBLEMS AS IS

# # Visualise training accuracy for El Nino & La Nina Events
# fig = plt.figure()
# plt.plot(lanina_train_accs, '-o', color='blue')
# plt.plot(elnino_train_accs, '-o', color='red')
# plt.plot(np.add(lanina_train_accs, elnino_train_accs).tolist(), '-o', color='black')
# plt.legend(['La Nina Accuracy', 'El Nino Accuracy', 'Combined Accuracy'], loc='lower right')
# plt.xlabel('epochs')
# plt.ylabel('accuracy in percent')
# # specify axis tick step sizes
# plt.xticks(np.arange(0, len(train_accuracies), 1))
# plt.yticks(np.arange(0, 105, 10))
# plt.title(f'Plot of extreme events train accuracy for {n_epochs} epochs')
# plt.show()
#
#
# # Visualise testing accuracy for El Nino & La Nina Events
# fig = plt.figure()
# plt.plot(lanina_test_accs, '-o', color='blue')
# plt.plot(elnino_test_accs, '-o', color='red')
# plt.plot(np.add(lanina_test_accs, elnino_test_accs).tolist(), '-o', color='black')
# plt.legend(['La Nina Accuracy', 'El Nino Accuracy', 'Combined Accuracy'], loc='lower right')
# plt.xlabel('epochs')
# plt.ylabel('accuracy in percent')
# # specify axis tick step sizes
# plt.xticks(np.arange(0, len(train_accuracies), 1))
# plt.yticks(np.arange(0, 105, 10))
# plt.title(f'Plot of extreme events test accuracy for {n_epochs} epochs')
# plt.show()