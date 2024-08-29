
import dask.distributed as dd
import xarray as xr
import dask.array as da
import numpy as np
import os
import zarr
from numcodecs import Blosc

import torch as tr
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from pftools.pipeline.core.microscope import MicroscopeInformation
from pftools.pipeline.core.algorithm import ParallelTileAnalysisTask
from pftools.pipeline.deconvolution.dw_jax import run_dw, run_dw_tiled
from pftools.pipeline.deconvolution.psf import generate_psf_custom

WAVELENGTHS = {
    'DAPI': 0.442,
    'FITC': 0.52,
    'Cy3': 0.59,
    'Cy5': 0.69,
    'Cy7': 0.82
}

@dataclass
class DeconvolveParams:
    n_iter:int=10
    method:str='shb'
    tile_factor:int=4
    compressor:Optional[str]='lz4'

class DWDeconvolve(ParallelTileAnalysisTask):
    def __init__(self, client: dd.Client, 
                 tiles:List[xr.DataArray], 
                 image_output_path:str,
                 output_path:str,
                 microscope:MicroscopeInformation,
                 params:Optional[DeconvolveParams]=None,
                 psfs:Optional[Dict[str, np.ndarray]]=None):
        super().__init__(client, tiles, output_path)
        self.logger.info("Generating PSFs")
        if psfs is None:
            self.psfs = self._generate_psfs()
        else:
            self.psfs = psfs
        self.image_output_path = image_output_path
        self.microscope = microscope
        if params is None:
            self.params = DeconvolveParams()
        else:
            self.params = params
        
    def _process_tile(self, tile_idx: int) -> None:
        curr_tile = self.tiles[tile_idx]
        # get the unique readout channels for this tile
        channels = np.unique(curr_tile.coords['readout_name'].values)
        # iterate over each unique readout channel
        for channel in channels:
            # get the color used by this channel
            chan_tile = curr_tile[curr_tile.coords['readout_name']==channel]
            # deconvolve the color channel
            tile = self.client.scatter(self.tiles[tile_idx]) 

            self.client.submit(self._deconvolve_color_channel, tile_idx, channel,)

    def _tile_processing_done_callback(self, tile_idx: int, future: dd.Future) -> None:
        """
        Callback for when a tile is done processing to save out data.
        """
        tile_idx, channel, img = future.result()
        # save out the deconvolved data
        self._save_deconvolved_data(tile_idx, channel, img)

    def _save_deconvolved_data(self, tile_idx:int, channel:str, img:np.ndarray) -> None:
        """
        Save out all rounds of deconvolved data for a tile to a single Zarr array, with compression. 
        """
        # get path for zarr file
        fpath = os.path.join(self.image_output_path, f"tile_{tile_idx}.zarr")
        # get compressor
        if self.params.compressor is not None:
            compressor = Blosc(cname=self.params.compressor, clevel=3, shuffle=Blosc.BITSHUFFLE)
        else:
            compressor = None
        # create zarr file if doesn't exist
        if not os.path.exists(fpath):
            file = zarr.open(fpath, mode='w')
        else:
            file = zarr.open(fpath, mode='a')
        
        # create a Zarr array for the data
        zdata = zarr.array(img, chunks=(1, img.shape[1], img.shape[2]), dtype=np.uint16, compressor=compressor)
        # set metadata for the array (channel and color)
        zdata.attrs['channel'] = channel
        #zdata.attrs['color'] = self.tiles[tile_idx].sel(channel=channel).color.values
        #zdata.attrs['xy_size'] = self.tiles[tile_idx].xy_size.values
        #zdata.attrs['z_size'] = self.tiles[tile_idx].z_size.values

        # save out the data
        file[channel] = zdata
        # try to free up some memory
        del img

    def _deconvolve_color_channel(self, tile_idx: int, channel: str) -> None:
        tile = self.tiles[tile_idx]
        psf = self.psfs[channel]
        img = tile.sel(channel=channel).values
        img = deconvolve_image(img, psf, n_iter=self.params.n_iter, method=self.params.method, tile_factor=self.params.tile_factor)
        return tile_idx, channel, img

    def _generate_psfs(self) -> Dict[str, np.ndarray]:
        """
        Generate PSFs for each channel.
        """
        psfs = {}
        for channel, wvl in WAVELENGTHS.items():
            psfs[channel] = self._generate_psf_for_wvl(wvl)
        return psfs

    def _generate_psf_for_wvl(self, wvl:float, xy_size:int=181,z_size:int=181) -> np.ndarray:
        """
        Generate a PSF for a given wavelength.
        """
        ni = self.microscope.ni
        na = self.microscope.na
        wd = self.microscope.wd
        tl = self.microscope.tubelens
        mag = self.microscope.mag
        dz = self.microscope.z_step
        dxy = self.microscope.microns_per_pixel
        return generate_psf_custom(dxy, dz, xy_size=xy_size, z_size=z_size,
                            M=mag, NA=na, n=1.33, wd=wd, tl=tl * 1.0e3, 
                            wvl=wvl, ni=ni)


def deconvolve_image(img: np.ndarray, psf: np.ndarray, n_iter:int=10, method:str='shb', tile_factor:int=1) -> np.ndarray:
    """
    Deconvolve a single tile using DW
    """
    # swap axes to XYZ
    img = np.swapaxes(img.astype(np.float32), 0, 2)

    # convert to torch
    #img = tr.from_numpy(img)
    #psf = tr.from_numpy(psf.astype(np.float32))
    # swap axes to XYZ
    psf = np.swapaxes(psf.astype(np.float32), 0, 2)

    # run DW
    if tile_factor == 1:
        img = run_dw(img, psf, n_iter=n_iter, method=method)
    else:
        # calculate tile size, assuming square tile
        tile_size = img.shape[-1] // tile_factor
        tile_overlap = int(tile_size * 0.1) # use 10% overlap
        img = run_dw_tiled(img, psf, n_iter=n_iter, method=method, 
                           tile_max_size=tile_size, tile_overlap=tile_overlap)

    #img = img.numpy().astype(np.uint16)
    img = img.astype(np.uint16)
    img = np.swapaxes(img, 0, 2) # swap back to ZYX
    return img
