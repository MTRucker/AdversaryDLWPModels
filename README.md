# **Adversarial Attacks on Deep Learning Weather Prediction Models**

Read this thoroughly before diving into the code, as some setup is required to run this code without any problems.

Bachelor's Thesis on Adversarial Attacks on Deep Learning Weather Prediction Models

Done within the MLCS Group at the University of Tübingen


## **WARNINGS for local users**
The dataset used is courtesy of their respective owners (ESGF, NCAR, the CMIP6 project and the CESM2 project).
The dataset can be found at the following links :
- At the original citation : https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.NCAR.CESM2.piControl
- At the CMIP 6 search interface : https://esgf-data.dkrz.de/search/cmip6-dkrz/ under the name CMIP6.CMIP.NCAR.CESM2.piControl.r1i1p1f1.Amon.tas.gn

Any other local files shouldn't require any external downloading. These could be accessed either by creating them with the code provided (models as an example) or by using someone's already files (examples are in this GitHub repository).


## **PACKAGES USED**
------------------------
- [PyTorch](https://pytorch.org/get-started/locally/) (CUDA optional, but helpful)
- [matplotlib](https://matplotlib.org/stable/users/getting_started/)
- [xarray](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html)
- [numpy](https://numpy.org/install/)
- [einops](https://einops.rocks/#Installation)
- [nc-time-axis](https://github.com/SciTools/nc-time-axis)
- preproc.py is a local script, but still necessary for constructing the dataset
