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
import statistics
import preproc
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
# current_adam_model = "drop"
current_adam_model = "nodrop"


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
        self.ds = xr.open_dataset(file)[var_label]

        # ----------------------------------------------
        # SHOW WHOLE WORLD AS SST + MASKED
        # rand_picked_time = random.randint(0, 14399)
        # world_plot = self.ds.isel(time=7862)
        # print(world_plot)
        # ax = plt.axes()
        # world_plot.plot(cmap="coolwarm")
        # ax.set_title('')
        # plt.show()
        #
        # f_lsm = "./data/sftlf_fx_CESM2_historical_r1i1p1f1.nc"
        # # Get lsm for loss masking of big cutout
        # land_area_mask = preproc.process_data(
        #     f_lsm, ['sftlf'],
        #     lon_range=(-180, 179),
        #     lat_range=(-90, 90),
        #     climatology=None,
        #     shortest=False,
        # )['sftlf']
        # print(land_area_mask)
        # lsm = th.from_numpy(land_area_mask.where(land_area_mask == 0, 1).data)
        # # set mask for big cutout
        # world_masked = world_plot.where(lsm == 0.0)
        # ax = plt.axes()
        # world_masked.plot(cmap="coolwarm")
        # ax.set_title('')
        # plt.show()
        #
        #
        # # NINO 3.4 CUTOUT ANOMALY
        # small_nino34_cutout = self.ds.sel(time=slice("0641-01-15", "0670-12-15"), lat=slice(-5, 5), lon=slice(-170, -120))
        # small_nino34_cutout = small_nino34_cutout.groupby('time.month') - small_nino34_cutout.groupby('time.month').mean(dim="time")
        # small_nino34_cutout = small_nino34_cutout.sel(time="0656-03-15")
        # small_nino34_cutout = th.tensor(small_nino34_cutout.data, dtype=th.float32)
        # fig = plt.figure()
        # plt.pcolor(small_nino34_cutout.squeeze(), cmap="coolwarm", vmin=-2, vmax=2)
        # plt.colorbar()
        # plt.show()
        #
        #
        # # BIG CUTOUT ANOMALY + MASKED
        # big_nino34_cutout = self.ds.sel(
        #     lon=self.ds.lon[(self.ds.lon < min(lon_range)) |
        #                     (self.ds.lon > max(lon_range))],
        #     lat=slice(np.min(lat_range), np.max(lat_range)))
        # big_nino34_cutout = big_nino34_cutout.roll(lon=39, roll_coords=True)
        # land_area_mask = preproc.process_data(
        #     f_lsm, ['sftlf'],
        #     lon_range=lon_range,
        #     lat_range=lat_range,
        #     climatology=None,
        #     shortest=True,
        # )['sftlf']
        # land_area_mask = land_area_mask.roll(lon=39, roll_coords=True)
        # lsm = th.from_numpy(land_area_mask.where(land_area_mask == 0, 1).data)
        # big_nino34_cutout_masked = big_nino34_cutout.where(lsm == 0.0)
        #
        # big_nino34_cutout_masked = big_nino34_cutout_masked.sel(time=slice("0641-01-15", "0670-12-15"))
        # big_nino34_cutout_masked = big_nino34_cutout_masked.groupby("time.month") - big_nino34_cutout_masked.groupby('time.month').mean(dim="time")
        # big_nino34_cutout_masked = big_nino34_cutout_masked.sel(time="0656-03-15")
        # big_nino34_cutout = big_nino34_cutout.sel(time = slice("0641-01-15", "0670-12-15"))
        # big_nino34_cutout = big_nino34_cutout.groupby("time.month") - big_nino34_cutout.groupby('time.month').mean(dim="time")
        # big_nino34_cutout = big_nino34_cutout.sel(time="0656-03-15")
        # big_nino34_cutout = th.tensor(big_nino34_cutout.data, dtype=th.float32)
        # fig = plt.figure()
        # plt.pcolor(big_nino34_cutout.squeeze(), cmap="coolwarm", vmin=-2, vmax=2)
        # plt.colorbar()
        # plt.show()
        # big_nino34_cutout_masked = th.tensor(big_nino34_cutout_masked.data, dtype=th.float32)
        # fig = plt.figure()
        # plt.pcolor(big_nino34_cutout_masked.squeeze(), cmap="coolwarm", vmin=-2, vmax=2)
        # plt.colorbar()
        # plt.show()
        #
        # # nino34 cutout
        # self.ds_nino34 = self.ds.sel(lat=slice(-5, 5), lon=slice(-170, -120))
        #
        # # normalize nino 34
        # self.normalizer = Normalizer(method='zscore')
        # self.ds_nino34 = self.normalizer.fit_transform(self.ds_nino34)
        # self.ds_nino34.attrs = {'normalizer': self.normalizer}
        # small_nino34_cutout = self.ds_nino34.isel(time=7862)
        # small_nino34_cutout = th.tensor(small_nino34_cutout.data, dtype=th.float32)
        # fig = plt.figure()
        # plt.pcolor(small_nino34_cutout.squeeze(), cmap="coolwarm")
        # plt.axis('off')
        # plt.show()
        #
        #
        # sys.exit()
        # ----------------------------------------------

        startingclock = time.time()
        climatology = self.ds.groupby('time.month').mean(dim='time')
        anomstart = self.ds.groupby('time.month') - climatology

        tsa_nino34 = anomstart.sel(lat=slice(-5, 5), lon=slice(-170, -120))
        nino34_index = tsa_nino34.mean(dim=['lat', 'lon'])
        nino34_index = nino34_index.rolling(time=5, center=True, min_periods=1).mean()
        print(f'Time it took to compute anomalies and nino 3.4 index : {time.time() - startingclock}')

        idx_lanina = np.where(nino34_index.data <= -0.5)[0]
        idx_elnino = np.where(nino34_index.data >= 0.5)[0]
        idx_neutral = np.where((nino34_index.data > -0.5) & (nino34_index.data < 0.5))[0]

        print('All Labeled Data Events within Dataset :')
        print(f'La Nina Events : [{len(idx_lanina)}/14400]{100. * len(idx_lanina) / 14400}')
        print(f'Neutral Events : [{len(idx_neutral)}/14400]{100. * len(idx_neutral) / 14400}')
        print(f'El Nino Events : [{len(idx_elnino)}/14400]{100. * len(idx_elnino) / 14400}\n')

        lanina_mean = anomstart.isel(time=idx_lanina).mean(dim='time')
        neutral_mean = anomstart.isel(time=idx_neutral).mean(dim='time')
        elnino_mean = anomstart.isel(time=idx_elnino).mean(dim='time')
        lanina_mean34 = tsa_nino34.isel(time=idx_lanina).mean(dim='time')
        neutral_mean34 = tsa_nino34.isel(time=idx_neutral).mean(dim='time')
        elnino_mean34 = tsa_nino34.isel(time=idx_elnino).mean(dim='time')

        fig = plt.figure(figsize=(14, 10))
        lanina_mean.plot(cmap='coolwarm')
        fig.suptitle("Mean of all categorized La Niña events")
        plt.show()

        fig = plt.figure(figsize=(14, 10))
        neutral_mean.plot(cmap='coolwarm')
        fig.suptitle("Mean of all categorized Neutral events")
        plt.show()

        fig = plt.figure(figsize=(14, 10))
        elnino_mean.plot(cmap='coolwarm')
        fig.suptitle("Mean of all categorized El Niño events")
        plt.show()

        fig = plt.figure(figsize=(14, 10))
        lanina_mean34.plot(cmap='coolwarm')
        fig.suptitle("Mean of all categorized La Niña events")
        plt.show()

        fig = plt.figure(figsize=(14, 10))
        neutral_mean34.plot(cmap='coolwarm')
        fig.suptitle("Mean of all categorized Neutral events")
        plt.show()

        fig = plt.figure(figsize=(14, 10))
        elnino_mean34.plot(cmap='coolwarm')
        fig.suptitle("Mean of all categorized El Niño events")
        plt.show()


        # tsa_bigcutout = anomstart.sel(
        #     lon=anomstart.lon[(anomstart.lon < min(lon_range)) |
        #                     (anomstart.lon > max(lon_range))],
        #     lat=slice(np.min(lat_range), np.max(lat_range)))
        # tsa_bigcutout = tsa_bigcutout.roll(lon=39, roll_coords=True)
        # print(tsa_bigcutout)
        #
        # lanina_mean_big = th.tensor(tsa_bigcutout.isel(time=idx_lanina).mean(dim='time').data, dtype=th.float32)
        # neutral_mean_big = th.tensor(tsa_bigcutout.isel(time=idx_neutral).mean(dim='time').data, dtype=th.float32)
        # elnino_mean_big = th.tensor(tsa_bigcutout.isel(time=idx_elnino).mean(dim='time').data, dtype=th.float32)

        # fig = plt.figure(figsize=(14, 10))
        # plt.pcolor(lanina_mean_big, cmap='coolwarm')
        # fig.suptitle("Mean of all categorized La Niña events")
        # plt.savefig(f"./plots_and_whatnots/lanina_plot{time_of_script_runs+2}.png")
        #
        # fig = plt.figure(figsize=(14, 10))
        # neutral_mean_big.plot(cmap='coolwarm')
        # fig.suptitle("Mean of all categorized Neutral events")
        # plt.savefig(f"./plots_and_whatnots/neutral_plot{time_of_script_runs+2}.png")
        #
        # fig = plt.figure(figsize=(14, 10))
        # elnino_mean_big.plot(cmap='coolwarm')
        # fig.suptitle("Mean of all categorized El Niño events")
        # plt.savefig(f"./plots_and_whatnots/elnino_plot{time_of_script_runs+2}.png")
        sys.exit()


        # select cutouts
        # big cutout -31,32 130,-70
        self.ds_big = self.ds.sel(
            lon=self.ds.lon[(self.ds.lon < min(lon_range)) |
                            (self.ds.lon > max(lon_range))],
            lat=slice(np.min(lat_range), np.max(lat_range)))
        # orders big cutout for easier plots
        self.ds_big = self.ds_big.roll(lon=39, roll_coords=True)

        # nino34 cutout
        self.ds_nino34 = self.ds.sel(lat=slice(-5, 5), lon=slice(-170, -120))


        # normalize nino 34
        startingclock = time.time()
        self.normalizer = Normalizer(method='zscore')
        self.ds_nino34 = self.normalizer.fit_transform(self.ds_nino34)
        self.ds_nino34.attrs = {'normalizer': self.normalizer}

        # normalize big cutout
        self.ds_big = self.normalizer.fit_transform(self.ds_big)
        self.ds_big.attrs = {'normalizer': self.normalizer}
        print(f'Time it took only to normalize cutouts : {time.time() - startingclock}')


        # MASKING
        f_lsm = "./data/sftlf_fx_CESM2_historical_r1i1p1f1.nc"
        # Get lsm for loss masking of big cutout
        land_area_mask = preproc.process_data(
            f_lsm, ['sftlf'],
            lon_range=lon_range,
            lat_range=lat_range,
            climatology=None,
        )['sftlf']
        land_area_mask = land_area_mask.roll(lon=39, roll_coords=True)
        lsm = th.from_numpy(land_area_mask.where(land_area_mask == 0, 1).data)
        # set mask for big cutout
        self.ds_big_masked = self.ds_big.where(lsm == 0.0)
        self._plot_mean(self.ds_big_masked, labels_list)
        #self._plot_mean(self.ds_nino34, labels_list)


        self.data = th.tensor(self.ds_big_masked.data, dtype=th.float32)
        #  plot first twelve months
        fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True)
        fig.suptitle("First twelve months, their classes and nino 3.4 index values")
        counter = 0
        for i in range(0, 4):
            for j in range(0, 3):
                temp_plot = axs[i, j].pcolor(self.data[counter], cmap="coolwarm", vmin=-2, vmax=2)
                fig.colorbar(temp_plot, ax=axs[i, j])
                if labels_list[counter] == 2:
                    axs[i, j].set_title("El Nino, {:.2f}".format(nino34_index[counter]))
                elif labels_list[counter] == 1:
                    axs[i, j].set_title("Neutral, {:.2f}".format(nino34_index[counter]))
                elif labels_list[counter] == 0:
                    axs[i, j].set_title("La Nina, {:.2f}".format(nino34_index[counter]))
                counter += 1
        plt.tight_layout()
        #plt.show()


        self.data = th.tensor(self.ds_nino34.data, dtype=th.float32)
        #  plot first twelve months
        fig, axs = plt.subplots(nrows=4, ncols=3, sharex=True, sharey=True)
        fig.suptitle("First twelve months, their classes and nino 3.4 index values")
        counter = 0
        for i in range(0, 4):
            for j in range(0, 3):
                temp_plot = axs[i, j].pcolor(self.data[counter], cmap="coolwarm", vmin=-2, vmax=2)
                fig.colorbar(temp_plot, ax=axs[i, j])
                if labels_list[counter] == 2:
                    axs[i, j].set_title("El Nino, {:.2f}".format(nino34_index[counter]))
                elif labels_list[counter] == 1:
                    axs[i, j].set_title("Neutral, {:.2f}".format(nino34_index[counter]))
                elif labels_list[counter] == 0:
                    axs[i, j].set_title("La Nina, {:.2f}".format(nino34_index[counter]))
                counter += 1
        plt.tight_layout()
        #plt.show()

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


            climatology = timeslice_30y.groupby('time.month').mean(dim='time')
            anom = timeslice_30y.groupby('time.month') - climatology

            # edge case so it doesn't do the last 30 years in smaller increments
            if time_end == "1201-01-05":
                for i in range(0, len(anom.data)):
                    labels_list.append(anom.data[i])
                break

            sliced_data = anom.data[0:12]
            for i in range(0, len(sliced_data)):
                labels_list.append(sliced_data[i])

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

    def _plot_mean(self, darray, labels_list):

        # get index of all elnino events
        elnino_indices = [i for i, x in enumerate(labels_list) if x == 2]
        # get index of all elnino events
        neutral_indices = [i for i, x in enumerate(labels_list) if x == 1]
        # get index of all elnino events
        lanina_indices = [i for i, x in enumerate(labels_list) if x == 0]

        # plot elnino mean
        ind_elnino = xr.DataArray(elnino_indices, dims=["time"])
        elnino_da = darray[ind_elnino]
        elnino_da = elnino_da.mean(dim="time")
        elnino_tensor = th.tensor(elnino_da.data, dtype=th.float32)
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle("The mean of all labeled El Nino Events")
        plt.pcolor(elnino_tensor, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar()
        plt.show()

        # plot neutral mean
        ind_neutral = xr.DataArray(neutral_indices, dims=["time"])
        neutral_da = darray[ind_neutral]
        neutral_da = neutral_da.mean(dim="time")
        neutral_tensor = th.tensor(neutral_da.data, dtype=th.float32)
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle("The mean of all labeled Neutral Events")
        plt.pcolor(neutral_tensor, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar()
        plt.show()

        # plot lanina mean
        ind_lanina = xr.DataArray(lanina_indices, dims=["time"])
        lanina_da = darray[ind_lanina]
        lanina_da = lanina_da.mean(dim="time")
        lanina_tensor = th.tensor(lanina_da.data, dtype=th.float32)
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle("The mean of all labeled La Nina Events")
        plt.pcolor(lanina_tensor, cmap="coolwarm", vmin=-2, vmax=2)
        plt.colorbar()
        plt.show()

    def __len__(self):
        return self.data.shape[1] - 1

    def __getitem__(self, idx):
        if idx + self.tau > (self.data.shape[1] - 1):
            return self.data[:, idx], self.nino_label_list[(self.data.shape[1] - 1)]
        return self.data[:, idx], self.nino_label_list[idx + self.tau]


# INPUTS FOR DATASET: FILEPATH, VAR_LABEL OF FILE, LAT OF CUTOUT, LON OF CUTOUT, MONTHS TO PREDICT#
start_time = time.time()
filename = "./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc"
nino34_dataset = ElNinoData(filename, 'ts', (-31, 32), (130, -70), tau_to_use)
# nino34_dataset = ElNinoData("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc", 'ts', (-5, 5), (-170, -120), tau_to_use)
end_time = time.time()
print(f'Time it took to prepare dataset : {end_time - start_time}')
sys.exit()
