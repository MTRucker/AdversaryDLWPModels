import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import sys
import numpy as np
import itertools
from torch.utils.data import Dataset, DataLoader
import preproc
import metric


class ElNinoData(Dataset):
    def __init__(self, file, var_label='ts', lat_range: tuple = (-90, 90), lon_range: tuple = (-180, 180)):
        '''
        file: path to netcdf file
        var_label: variable label in netcdf file
        lat_range: latitude range for data
        lon_range: longitude range for data
        '''
        # create lat lon slices
        latitudes = slice(lat_range[0], lat_range[1])
        longitudes = slice(lon_range[0], lon_range[1])
        # open dataset
        self.ds = xr.open_dataset(file)

        # data preprocessing
        # cuts dataset into specified lon and lat range
        self.ds = self.ds.sel(lat=latitudes, lon=longitudes)[var_label]
        # calculates means and deviation of cutout (can input time_rolling for a rolling mean)
        self.ts_mean, self.ts_std = self._compute_mean(darray=self.ds, time_roll=0)
        # compute anomalies/nino34 indices
        self.nino_label_list = self._compute_nino_idcs(ts_mean=self.ts_mean, ts_std=self.ts_std)

        # create data tensor
        self.data = torch.tensor(self.ds.data)

    # snippet taken from preproc.py's get_mean_time_series function
    def _compute_mean(self, darray, time_roll=0):

        ts_mean = darray - darray.mean(dim='time')
        ts_mean = ts_mean.mean(dim=['lon', 'lat'])
        if time_roll > 0:
            ts_mean = ts_mean.rolling(time=time_roll, center=True).mean()
        ts_std = darray.std()

        return ts_mean, ts_std

    # normalizes data and labels month as El Nino (1), Neutral (0) or La Nina (-1)
    def _compute_nino_idcs(self, ts_mean, ts_std):

        normalized_data = ts_mean
        labels_list = []

        for i in range(0, normalized_data['time'].size):
            current_nino34_value = normalized_data.isel(time=i).data

            # If > 0.5 degrees Celsius/Kelvin, then El Nino
            if current_nino34_value > 0.5:
                labels_list.append(1)

            # If < -0.5 degrees Celsius/Kelvin, then La Nina
            elif current_nino34_value < -0.5:
                labels_list.append(-1)

            # Else neutral (even for missing values)
            else:
                labels_list.append(0)

        return labels_list


    def __len__(self):
        return self.data.shape[0] - 1

    def __getitem__(self, idx):
        # change typeof to either get a PyTorch Tensor or an Xarray Datarray
        # currently set to datarray for easier plots , but tensor makes more sense in the future
        typeof = 'datarray'
        if typeof == 'tensor':
            return self.data[idx], self.nino_label_list[idx]
        elif typeof == 'datarray':
            return self.ds.isel(time=idx), self.nino_label_list[idx]


dataset = ElNinoData("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc", 'ts', (-5,  5),  (-170, -120))

# example from PyTorch tutorial to visualize dataset
labels_map = {
    0: "Neutral",
    1: "El Niño",
    -1: "La Niña",
}
fig = plt.figure(figsize=(14,10))
cols, rows = 2, 2
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(dataset),  size=(1,)).item()
    img, label = dataset[sample_idx]
    fig.add_subplot(rows, cols, i)
    img.plot(cmap='coolwarm')
    plt.title(f"{labels_map[label]}, {img['time'].data.item().strftime()[0:7]}")
plt.show()


# --------------------------------------------------------------------------
# REMNANTS OF PREVIOUS CODE, MAY COME IN HANDY
# OPEN DATASET
# da = xr.open_dataset("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc")
# # CREATE NINO 3.4 NC FILE AND THEIR INDICES/LABELS
# print(da)
# nino34_region = da['ts'].sel(lat=slice(-5, 5), lon=slice(-170, -120)).copy(deep=True)
# print(nino34_region.isel(time=0).data)
# print(f'\n\n{nino34_region.data[0]}')
# sys.exit()
# ds = xr.Dataset()
#
# # first and last year of dataset  (determined manually, written as string for ease of access)
# time_start_year = 1
# time_end_year = 1200
# time_step_size = 30
#
# # iterate over all years in 30 year steps ("Usually the anomalies are computed relative to a base period of 30 years.")
# for x in range(time_start_year, time_end_year - 28, time_step_size):
#
#     # Code to fill the string to the 0001-1200 year format
#     time_start = str(x)
#     start_len = len(time_start)
#     time_end = str(x + 30)
#     end_len = len(time_end)
#
#     if start_len < 4:
#         while(start_len < 4):
#             time_start = "0" + time_start
#             start_len = len(time_start)
#
#     if end_len < 4:
#         while (end_len < 4):
#             time_end = "0" + time_end
#             end_len = len(time_end)
#
#     time_start = time_start + "-12-05"
#     time_end = time_end + "-12-05"
#
#     # Calculate mean of Nino 3.4 Index and save within the new dataset
#     nino34 = nino34_region - nino34_region.mean(dim='time')
#     da['ts'].data =  nino34.data
# --------------------------------------------------------------------------