# **Adversarial Attacks on Deep Learning Weather Prediction Models**

Bachelor's Thesis on Adversarial Attacks on Deep Learning Weather Prediction Models

Done within the MLCS Group at the University of Tübingen



Read this thoroughly before diving into the code, as some setup is required to run this code without any problems.

I've also left some remnant code, images and data, because they have potential solutions to questions that may arise, but they are not essential. They have been left uncleaned as is, but I'm always available if need be and if some things need to be cleared up.


## **WARNINGS for local users**
The dataset used is courtesy of their respective owners (ESGF, NCAR, the CMIP6 project and the CESM2 project).
The dataset can be found at the following links :
- At the original citation : https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.NCAR.CESM2.piControl
- At the CMIP 6 search interface : https://esgf-data.dkrz.de/search/cmip6-dkrz/ under the name CMIP6.CMIP.NCAR.CESM2.piControl.r1i1p1f1.Amon.tas.gn

Any other local files shouldn't require any external downloading. These could be accessed either by creating them with the code provided (models as an example) or by using someone's already files (examples are in this GitHub repository).

Most of the code was run locally and not on the ML Cluster, as I had some issues setting everything up there.
However this should run completely fine on the cluster with the appropriate packages.


## **PACKAGES USED**
------------------------
- [PyTorch](https://pytorch.org/get-started/locally/) (CUDA optional, but helpful)
- [matplotlib](https://matplotlib.org/stable/users/getting_started/)
- [xarray](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html)
- [numpy](https://numpy.org/install/)
- [einops](https://einops.rocks/#Installation)
- [nc-time-axis](https://github.com/SciTools/nc-time-axis)
- [preproc.py](../main/preproc.py) is a local script, but still necessary for constructing the dataset


## GENERAL WORKFLOW
1. Train a Niño 3.4 Model by running nino34_model_training.py
   1. Prepares dataset; calculate Niño 3.4 cutout, Index, anomalies and labels
   2. Constructs model; see **MODEL ARCHITECTURE** below
   3. Set Hyperparameters (points i and ii are for clarification, the code does that "itself")
   4. Train and test model, model saves during training
   5. Visualize training and testing results as you want
2. Execute Adversarial Attack on Trained Model, by running nino34_adversarial_attack.py
   1. Set correct path to trained model
   2. Set Hyperparameters (magnitudes of adversarial attacks)
   3. Execute adversarial attack
   4. Visualize results as you want

## MODEL ARCHITECTURE
1. 1x Convolutional (Conv2d) layer : Projects the input to the chosen (128) latent dimension with a simple 1x1 convolution
2. 4x ConvNeXt blocks : Processes the spatial information with a series of Residual blocks, that are structured as follows;
   - 1x Conv2d layer : Process spatial information per channel with a 7x7 convolution, wherein each input channel is convolved with its own set of filters (size 1 in our case)
   - 1x Group normalization (GN) layer : Normalize each channel to have mean of 0 and standard deviation of 1, this stabilizes training
   - 1x Conv2d layer : Expand to higher dimension with an expansion factor of 4 (4 · 128 = 512) and a 1x1 convolution
   - 1x Gaussian Error Linear Units (GELU) activation function: Applies non-linearity
   - 1x Conv2d layer : Project back to lower dim (128) with a 1x1 convolution
3. 1x Global Average Pooling layer : Compute the average of all pixels in each feature map, return that as a single element which building block of the vector of all pooled feature maps
4. 1x Linear layer : Expand to higher dimension with an expansion factor of 4 (4 · 128 = 512)
5. 1x GELU activation function : Applies non-linearity
6. 1x Linear layer : Final classification layer, reduces higher dimension to dimension equal to desired amount of classes
![alt text](https://github.com/MTRucker/AdversaryDLWPModels/blob/main/Model_Architecture.png)
