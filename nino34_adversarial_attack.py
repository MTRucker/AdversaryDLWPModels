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
# This also affects the path of loading the model for attacking
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
# SETTING UP MODEL TO BE USED DURING ATTACK

# Setting Hyperparameters
batch_size_test = 1
learning_rate = 0.01


# Initializing Dataloaders for running adversarial attack
test_loader = DataLoader(test_data, batch_size=batch_size_test, shuffle=True)


# Initializing Network and Optimizer (ADAMW with exponential learning rate scheduler)
network = Nino_classifier()
network = network.to(device)
optimizer = optim.AdamW(network.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.90)


# Load Model (and Optimizer) to be tested
network_state_dict = th.load(f'./results/adamw_tau{tau_to_use}_nino34_model.pth')
# This line below is optional, but I needed it, as it allowed me to unpack CUDA saved models. My advice; keep it
network_state_dict = {key.replace("module.", ""): value for key, value in network_state_dict.items()}
network.load_state_dict(network_state_dict)
optimizer_state_dict = th.load(f'./results/adamw_tau{tau_to_use}_nino34_optimizer.pth')
optimizer.load_state_dict(optimizer_state_dict)
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# DEFINING ADVERSARIAL ATTACK FUNCTION
def adv_attack(model, use_device, curr_loader, eps):
    '''
    model: NN/Model to be used for testing
    use_device: Device on which all tensors should be used
    curr_loader: DataLoader Object used for testing
    eps: int, for determining magnitudes of perturbations
    '''
    model.eval()
    correct = 0
    adv_examples = []

    # Iterate over every datapoint in the "testing" DataLoader
    for data, target, anom in curr_loader:

        data, target = data.to(use_device), target.to(use_device)

        # Set requires_grad attribute of tensor. Important for attack, because gradients required
        data.requires_grad = True

        output = model(data)
        # Get model's initial prediction
        initial_pred = output.max(1, keepdim=True)[1]
        # Ignore attack if prediction wrong
        if initial_pred.item() != target.item():
            continue


        # ------------------------
        # ADVERSARIAL ATTACK HERE
        # We are using the Fast Gradient Sign Method (FGSM)

        # Set gradients to zero, because PyTorch accumulates them per default
        model.zero_grad()
        # Cross entropy loss
        loss = F.cross_entropy(output, target)
        # Backpropagation, for calculating gradients of loss function
        loss.backward()

        # Calculate perturbed image according to FGSM
        perturbed_image = data + (eps * data.grad.data.sign())
        # Extract perturbation by simply substracting perturbed image from original image
        perturbation = data - perturbed_image
        # ------------------------


        output = model(perturbed_image)
        # Get model's new prediction, based on manipulated data
        final_pred = output.max(1, keepdim=True)[1]


        # Check if attack unsuccessful (else save some examples)
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


    # calculating loss and accuracy of test/adversarial attack
    test_accuracy = 100. * correct / len(curr_loader.dataset)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {:.2f}%".format(
        eps, correct, len(curr_loader) * batch_size_test, test_accuracy))
    return test_accuracy, adv_examples


# THIS IS ONLY FOR SHOWING ONE IMAGE WITH SCALING EPSILONS
# This function is the preferred visualisation of adversarial attacks
def onetest(model, use_device, curr_loader, eps):
    '''
    model: NN/Model to be used for testing
    use_device: Device on which all tensors should be used
    curr_loader: DataLoader Object used for testing
    eps: list of int, for determining magnitude of perturbation
    '''
    model.eval()
    adv_examples = []
    anom_examples = []
    confidences = []

    for data, target, anom in curr_loader:

        data, target, anom = data.to(use_device), target.to(use_device), anom.to(use_device)

        # set requires_grad attribute of tensor. Important for attack, because gradients required
        data.requires_grad = True
        output = model(data)

        # Get model's initial confidence
        init_probs = F.softmax(output, dim=-1)
        init_conf = [i for i in init_probs.tolist()[0]]
        init_conf = max(init_conf)
        init_conf = round(init_conf * 100, 2)

        # Keep fishing for adversarial attacks that have a smaller initial confidence (<95%)
        if init_conf > 95.0:
            continue
        else:

            # Get model's intial prediction (redundant)
            initial_pred = output.max(1, keepdim=True)[1]
            # Ignore attack if prediction wrong
            if initial_pred.item() != target.item():
                continue


            # ------------------------
            # ADVERSARIAL ATTACK HERE (pretty much copied from before)

            # Set gradients to zero, because PyTorch accumulates them per default
            model.zero_grad()
            # Cross entropy loss
            loss = F.cross_entropy(output, target)
            # Backpropagation, for calculating gradients of loss function
            loss.backward()

            # Calculate perturbed image according to FGSM (this time it immediately starts with eps = 0.05)
            perturbed_image = data + (eps[1] * data.grad.data.sign())
            # Extract perturbation by simply substracting perturbed image from original image
            perturbation = data - perturbed_image
            # ------------------------


            output = model(perturbed_image)
            # Get model's new prediction, based on manipulated data
            final_pred = output.max(1, keepdim=True)[1]


            # Check if attack successful with 0.05 epsilon
            if final_pred.item() != target.item():

                # If attack succesfull, run through all epsilons and then return this one attempt
                for i in range(0, len(eps)):

                    output = model(data)

                    # Get model's initial confidence
                    init_probs = F.softmax(output, dim=-1)
                    init_conf = [i for i in init_probs.tolist()[0]]
                    init_conf = max(init_conf)
                    init_conf = round(init_conf * 100, 2)

                    # Get model's prediction
                    initial_pred = output.max(1, keepdim=True)[1]


                    # ------------------------
                    # ADVERSARIAL ATTACK HERE

                    # Set gradients to zero, because PyTorch accumulates them per default
                    model.zero_grad()
                    # Cross entropy loss
                    loss = F.cross_entropy(output, target)
                    # Backpropagation, for calculating gradients of loss function
                    loss.backward()

                    # Calculate perturbed image according to FGSM
                    perturbed_image = data + (eps[i] * data.grad.data.sign())
                    # Calculate perturbed anomaly, purely for visuals, and not really relevant
                    perturbed_anom = anom + (eps[i] * data.grad.data.sign())
                    # Extract perturbation by simply substracting perturbed image from original image
                    perturbation = data - perturbed_image
                    # ------------------------


                    output = model(perturbed_image)
                    # Get model's new prediction, based on manipulated data
                    final_pred = output.max(1, keepdim=True)[1]

                    # Get model's final confidence
                    final_probs = F.softmax(output, dim=-1)
                    final_conf = [i for i in final_probs.tolist()[0]]
                    final_conf = max(final_conf)
                    final_conf = round(final_conf * 100, 2)


                    output = model(perturbation)
                    # Get model's prediction of solely the perturbation
                    pert_pred = output.max(1, keepdim=True)[1]

                    # Get model's confidence on perturbation
                    pert_probs = F.softmax(output, dim=-1)
                    pert_conf = [i for i in pert_probs.tolist()[0]]
                    pert_conf = max(pert_conf)
                    pert_conf = round(pert_conf * 100, 2)


                    # Add confidences and predicted class of perturbation to list
                    confidences.append((init_conf, pert_conf, final_conf, pert_pred.item()))


                    # Save original image (+ prediction), perturbation and perturbed image (+ prediction) at current epsilon
                    adv_ex = perturbed_image.squeeze().detach().cpu().numpy()
                    pert = perturbation.squeeze().detach().cpu().numpy()
                    corr_ex = data.squeeze().detach().cpu().numpy()
                    temp_list = [(initial_pred.item(), final_pred.item(), corr_ex, pert, adv_ex)]
                    adv_examples.append(temp_list)

                    # Save anomalies + perturbed anomalies as well
                    corr_anom = anom.squeeze().detach().cpu().numpy()
                    adv_anom = perturbed_anom.squeeze().detach().cpu().numpy()
                    temp_list = [(corr_anom, adv_anom)]
                    anom_examples.append(temp_list)

                break


    return adv_examples, anom_examples, confidences
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# TESTING ADVERSARIAL ATTACK

# Set magnitudes of epsilons (7 epsilons recommended, as they work with the plots below)
epsilons = [0, .05, .1, .15, .2, .25, .3]

# TRACKERS to use later for analysis and printouts
test_accuracies = []
examples = []


# RUNNING THE ATTACK
for eps in epsilons:
    acc, ex = adv_attack(network, device, test_loader, eps)
    test_accuracies.append(acc)
    examples.append(ex)


# THIS IS ONLY FOR SHOWING ONE IMAGE WITH SCALING EPSILONS
# This function yields easy to visualise examples
new_examples, anoms, confs = onetest(network, device, test_loader, epsilons)
# ----------------------------------------------------------------------




# ----------------------------------------------------------------------
# PLOTS

# OPTIONAL : write epsilon-accuracies to csv file to later plot a nice overlapping plot between multiple models
#            (see epsilon_reader_nino34.py)
# with open('./adversarial/nino34_epsilon_accuracies.csv', 'a', encoding='UTF-8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(test_accuracies)


# Visualise new validation accuracy
fig = plt.figure()
plt.plot(test_accuracies, '-o', color='blue')
plt.legend(['Test Accuracy'], loc='lower right')
plt.xlabel('epsilons')
plt.ylabel('accuracy in percent')
plt.xticks(range(len(test_accuracies)), epsilons)
plt.yticks(np.arange(0, 105, 10))
plt.title('Accuracy vs Epsilon')
plt.show()




# ALL PLOTS BELOW ARE PLOTTED WITH A CONSTRAINED COLORBAR [-2 - 2]
# FOR STANDARDIZATION AND EASIER VISUALIZATION


# Helper function for plots below (it gives a name back for an inserted number)
def name_my_event(number: int = 0):
    if number == 0:
        name = "La Niña"
    elif number == 1:
        name = "Neutral"
    elif number == 2:
        name = "El Niño"
    return name


# Plot each epsilon example with a
# 1. original image, initial prediction and initial confidence (only once, as it remains unchanged)
# 2. perturbation, perturbation prediction and perturbation confidence
# 3. final perturbed image, perturbed image prediction and perturbed image confidence
cnt = 0
plt.figure()
for i in range(len(epsilons)):
    for j in range(len(examples[i])):

        # Get predictions, confidences, images for current epsilon
        orig, adv, corr_ex, pert, adv_ex = examples[i][j]
        conf1, conf2, conf3, pert_pred = confs[i]

        cnt += 1


        # Create subplot 1 in current row and "delete" its axes (subplot 1; original image)
        plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        # If first subplot in row, then name according to epsilon
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)

        # If first subplot, then display original prediction and confidence (only once)
        if cnt <= 1:
            plt.title(f"Original Image, Class : {name_my_event(orig)}") # name whole column
            plt.xlabel(f"Predicted Class : {name_my_event(orig)}, {conf1}% Confidence")

        plt.pcolor(corr_ex, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar()

        cnt += 1


        # Create subplot 2 in current row and "delete" its axes (subplot 2; perturbation)
        plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])

        if cnt <= 2:
            plt.title("Perturbation") # name whole column
        # Attach prediction and confidence to current subplot
        plt.xlabel(f"Predicted Class : {name_my_event(pert_pred)}, {conf2}% Confidence")

        plt.pcolor(pert, cmap="coolwarm")

        cnt += 1


        # Create subplot 3 in current row and "delete" its axes (subplot 3; perturbed image)
        plt.subplot(len(epsilons), len(examples[0]) + 2, cnt)
        plt.xticks([], [])
        plt.yticks([], [])

        if cnt <= 3:
            plt.title("Perturbed Image") # name whole column
        # Attach prediction and confidence to current subplot
        plt.xlabel(f"Predicted Class : {name_my_event(adv)}, {conf3}% Confidence")

        plt.pcolor(adv_ex, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar()
# plt.tight_layout()
plt.show()


# Plot each epsilon example with a
# 1. original anomaly, initial prediction and initial confidence (only once, as it remains unchanged)
# 2. final perturbed anomaly, perturbed IMAGE prediction and perturbed IMAGE confidence
# all predictions and confidences (identical to above plot) are based upon the original image, not the anomalies
cnt = 0
plt.figure()
for i in range(len(epsilons)):
    for j in range(len(anoms[i])):

        # Get predictions, confidences, images and anomalies for current epsilon
        orig, adv, corr_ex, pert, adv_ex = examples[i][j]
        conf1, conf2, conf3, pert_pred = confs[i]
        corr_anom, adv_anom = anoms[i][j]

        cnt += 1


        # Create subplot 1 in current row and "delete" its axes (subplot 1; original anomaly)
        plt.subplot(len(epsilons), len(anoms[0]) + 1, cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        # If first subplot in row, then name according to epsilon
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)

        # If first subplot, then display original prediction and confidence (only once)
        if cnt <= 1:
            plt.title("Original Anomaly") # name whole column
            plt.xlabel(f"Predicted Class : {name_my_event(orig)}, {conf1}% Confidence")


        # Plotted with constrained colorbar, for standardization and easier visualisation
        plt.pcolor(corr_anom, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar()

        cnt += 1


        # Create subplot 2 in current row and "delete" its axes (subplot 2; perturbed anomaly)
        plt.subplot(len(epsilons), len(anoms[0]) + 1, cnt)
        plt.xticks([], [])
        plt.yticks([], [])

        # Visual flare; add name to first perturbed anomaly in column
        if cnt <= 2:
            plt.title("Perturbed Anomaly") # name whole column

        # Display prediction and confidence for perturbed image (NOT ANOMALY, BUT NORMAL PERTURBED IMAGE)
        plt.xlabel(f"Predicted Class : {name_my_event(adv)}, {conf3}% Confidence")

        # Plotted with constrained colorbar, for standardization and easier visualisation
        plt.pcolor(adv_anom, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar()
plt.tight_layout()
plt.show()