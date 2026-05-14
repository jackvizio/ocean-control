import xarray as xr
ds = xr.open_dataset("hydro_coefficients.nc")
print(ds.data_vars)