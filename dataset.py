import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import sys
import numpy as np
import itertools
from torch.utils.data import Dataset, DataLoader
import nc_time_axis
from preproc import Normalizer

# --------------------------------------------------------------------------
# REMNANTS OF PREVIOUS CODE, MAY COME IN HANDY
# OPEN DATASET
# da = xr.open_dataset("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc")
# # CREATE NINO 3.4 NC FILE AND THEIR INDICES/LABELS
# print(da)
# nino34_region = da['ts'].sel(lat=slice(-5, 5), lon=slice(-170, -120))
# print(da['time'].data.min())
# print(da['time'].data.max())

class ElNinoData(Dataset):
    def __init__(self, file, var_label='ts', lat_range: tuple = (-90, 90), lon_range: tuple = (-180, 180), tau: int = 1):
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
        self.ds = self.ds.sel(lat=latitudes, lon=longitudes)[var_label]
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
        self.data = torch.tensor(self.ds.data)

    def _compute_anomalies_nino34(self, darray):

        # list of all computed nino34 years and months
        labels_list = []

        # first and last year of dataset
        time_start_year = darray['time'].data.min().year
        time_end_year = darray['time'].data.max().year
        time_step_size = 1

        # iterate over all years
        for x in range(time_start_year, time_end_year + time_step_size, time_step_size):

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
            anom = timeslice_30y - timeslice_30y.mean(dim='time')
            nino34 = anom.mean(dim=['lat', 'lon'])
            sliced_data = nino34.data[0:12]
            for i in range(0, len(sliced_data)):
                labels_list.append(sliced_data[i])

        return labels_list

    def _label_data(self, darray, anomalies):

        # list of all labeled nino34 events within time span (1200 years)
        labels_list =  []

        for i in anomalies:
            if i > 0.5:
                labels_list.append(1)
            if i < -0.5:
                labels_list.append(-1)
            else:
                labels_list.append(0)

        return labels_list

    def __len__(self):
        return self.data.shape[0] - 1

    def __getitem__(self, idx):
        # change typeof to either get a PyTorch Tensor or an Xarray Datarray
        # currently set to datarray for easier plots , but tensor makes more sense in the future
        typeof = 'tensor'
        if typeof == 'tensor':
            return self.data[idx], self.nino_label_list[idx+self.tau]
        elif typeof == 'datarray':
            return self.ds.isel(time=idx), self.nino_label_list[idx+self.tau]


# INPUTS : FILEPATH, VAR_LABEL OF FILE, LAT OF CUTOUT, LON OF CUTOUT, MONTHS TO PREDICT
dataset = ElNinoData("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc", 'ts', (-5,  5),  (-170, -120), 3)

# # example from PyTorch tutorial to visualize dataset
# labels_map = {
#     0: "Neutral",
#     1: "El Niño",
#     -1: "La Niña",
# }
# fig = plt.figure(figsize=(14,10))
# cols, rows = 2, 2
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(dataset),  size=(1,)).item()
#     img, label = dataset[sample_idx]
#     fig.add_subplot(rows, cols, i)
#     img.plot(cmap='coolwarm')
#     plt.title(f"{labels_map[label]}, {img['time'].data.item().strftime()[0:7]}")
# plt.show()