import napari
import tifffile
import xarray as xr
import numpy as np

from pftools.pipeline.core.tile import unroll_tile

def load_zscan_4cols(fpath):
    """
    Load 4 color image stack with interleaved frames.
    Reshape into 4 x nz x nx x ny stack
    """
    with tifffile.TiffFile(fpath) as tif:
        data = tif.asarray()
        nz, nx, ny = data.shape
        data = np.reshape(data, (nz//4, 4, nx, ny))
        data = np.swapaxes(data, 0, 1)
    return data

def load_zscan_5cols(fpath):
    """
    Load 5 color image stack with interleaved frames.
    Reshape into 5 x nz x nx x ny stack
    """
    with tifffile.TiffFile(fpath) as tif:
        data = tif.asarray()
        nz, nx, ny = data.shape
        data = np.reshape(data, (nz//5, 5, nx, ny))
        data = np.swapaxes(data, 0, 1)
    return data

def view_zscan_5cols(fpath, contrast_lims=[0, 65535], max_projection=False):
    """
    Load 5 color image stack with interleaved frames.
    Reshape into 5 x nz x nx x ny stack
    """
    data = load_zscan_5cols(fpath)
    viewer = napari.Viewer()
    for i in range(5):
        if max_projection:
            viewer.add_image(np.max(data[i], axis=0), name=f'ch{i}', colormap='gray', contrast_limits=contrast_lims)
        else:
            viewer.add_image(data[i], name=f'ch{i}', colormap='gray', contrast_limits=contrast_lims)

def visualize_tile(tile:xr.DataArray, unroll:bool=False, contrast_lims=[0,65535]):
    if unroll:
        t = unroll_tile(tile, coords=['readout_name', 'z'], swap_dims=True, rechunk=False)
        # get dask array
        t = t.data.astype(np.uint16)
    else:
        t = tile
    viewer = napari.Viewer()
    viewer.add_image(t, contrast_limits=contrast_lims) 
    return viewer