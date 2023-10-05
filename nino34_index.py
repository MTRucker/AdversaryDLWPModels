import cftime
import numpy as np
import xarray as xr
import matplotlib as mpl
import nc_time_axis
import matplotlib.pyplot as plt
import sys
# ----------------------------------------------
# OPEN DATASET
da = xr.open_dataset("./data/ts_Amon_CESM2_piControl_r1i1p1f1.nc")
for x, y in da.attrs.items():
    print(f'{x} : {y}')
print(f'\n\n {da}')
# print(da.attrs)
ts_set = da['ts']

# ----------------------------------------------
# PLOT WORLD AT FIRST AND LAST INSTANCE
fig, axs = plt.subplots(2, 1, figsize=(14, 10))
for i, t in enumerate(['0001-01', '1200-12']):
    ts_set.sel(time=t).plot(ax=axs[i], cmap='coolwarm')
plt.show()

# ----------------------------------------------
# PLOT NINO 3.4 ON MAP AT FIRST AND LAST INSTANCE
nino34_region_original = ts_set.sel(lat=slice(-5, 5), lon=slice(-170, -120))
fig, axs = plt.subplots(2, 1, figsize=(14, 10))
for i, t in enumerate(['0001-01', '1200-12']):
    nino34_region_original.sel(time=t).plot(ax=axs[i], cmap='coolwarm')
plt.show()

# ----------------------------------------------
# PLOT NINO 3.4 INDEX
# Set time slice
time_slice_start = '0001-12-05'
time_slice_end = '0031-12-05'
nino34_region = ts_set.sel(time=slice(time_slice_start, time_slice_end), lat=slice(-5, 5), lon=slice(-170, -120))

# Calculate mean and rolling mean
nino34 = nino34_region - nino34_region.mean(dim='time')
nino34 = nino34.mean(dim=['lat', 'lon'])
nino34_rolling_mean = nino34.rolling(time=5, center=True).mean()
# Calculate standard deviation
std_dev = nino34_region.std()
# Normalize data
nino34_rolling_mean = nino34_rolling_mean / std_dev

fig, ax = plt.subplots(figsize=(14, 10))
fig.suptitle(f'Visualization of Niño 3.4 Index between the years {time_slice_start[0:4]} and {time_slice_end[0:4]}')

# Get highest and lowest recorded Nino 3.4 Index
x_max = nino34_rolling_mean.isel(time=nino34_rolling_mean.argmax()).time.data
y_max = nino34_rolling_mean.max().data
x_max_date = x_max.item().strftime()[0:7]
x_min = nino34_rolling_mean.isel(time=nino34_rolling_mean.argmin()).time.data
y_min = nino34_rolling_mean.min().data
x_min_date = x_min.item().strftime()[0:7]

print(nino34_rolling_mean.data)
print(type(nino34_rolling_mean.data))
print(nino34_rolling_mean.data.shape)
sys.exit()

ax.plot(nino34_rolling_mean['time'].data, nino34_rolling_mean.data, '-k')
ax.fill_between(nino34_rolling_mean['time'].data, nino34_rolling_mean.data, y2=0.5, where=nino34_rolling_mean.data >= 0.5, color='r', alpha=0.8)
ax.fill_between(nino34_rolling_mean['time'].data, nino34_rolling_mean.data, y2=-0.5, where=nino34_rolling_mean.data <= -0.5, color='b', alpha=0.8)
ax.axhline(-0.5, color='k')
ax.axhline(0.5, color='k')
ax.annotate(x_max_date, xy=(x_max, y_max + y_max/20))
ax.annotate(x_min_date, xy=(x_min, y_min + y_min/20))
ax.set_ylabel('Nino3.4 index')
plt.show()

# ----------------------------------------------
# PLOT THE 2 EXTREMES WITHIN SELECTED TIME SLICE
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Visualizations of max El Niño and min La Niña from years {time_slice_start[0:4]} to {time_slice_end[0:4]}'
             f'\nHighest Temp Date : {x_max_date}\nLowest Temp Date : {x_min_date}')
for i, t in enumerate([x_max, x_min]):
    nino34_region_original.sel(time=t).plot(ax=axs[i,  0], cmap='coolwarm')
    (nino34_region - nino34_region.mean(dim='time')).sel(time=t).plot(ax=axs[i, 1], cmap='coolwarm')

axs[0,0].set_title(f'Highest Temperature Anomaly')
axs[0,1].set_title(f'Highest Temperature Anomaly, averaged')
axs[1,0].set_title(f'Lowest Temperature Anomaly')
axs[1,1].set_title(f'Lowest Temperature Anomaly, averaged')

plt.show()