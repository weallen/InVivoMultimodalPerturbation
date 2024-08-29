from __future__ import annotations
import os
import functools
from typing import List, Callable, Optional, Tuple
import numpy as np
import dask.array as da
import dask.distributed as dd
import pandas as pd
import dask_image.imread as imread
import xarray as xr
import zarr
from tqdm import tqdm
from pftools.pipeline.core.dataorganization import DataOrganization
from pftools.pipeline.core.dataset import ExperimentData
from pftools.pipeline.util.utils import unique_unsorted
from pftools.pipeline.util.preprocessing import imagestack_transpose, imagestack_flipud, imagestack_fliplr

from pftools.pipeline.processing.registration import register_tile_by_image_index 
# These classes are responsible for actually (lazily) loading the data
# from disk. They are used by the Experiment class.

# A Tile is a xr.DataArray containing the images for a single FOV. 
# It was easier to not create a wrapper class, so as not to have to wrap all the methods of xr.DataArray.
# It can either have dimensions Nz x Nchannel x Nx x Ny or (Nchannel*Nz) x Nx x Ny
# Tiles are lazily loaded by a TileLoader, using the dataset and data organization interfaces.

class TileLoader(object):
    """
    A lightweight object to wrap loading an xr.DataArray.
    Knows how the data is organized, and can access through the dataset interface based on the data organization.

    This allows us to push down the actual data loading to the client, so that we can limit transfers of data over the network
    and only load the data when we need it.
    """   
    def __init__(self, fov:int, dorg:DataOrganization, dataset:ExperimentData, 
                 flip_horizontal:bool=False, flip_vertical:bool=False, transpose:bool=True,
                 ref_idx:int=0, readouts:Optional[List[str]]=None):

        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.transpose = transpose
        self.dataset = dataset
        self.dorg = dorg
        self.data = None
        self.fov = fov
        self.ref_idx = ref_idx
        pos = dataset.get_positions()
        self.position = pos[pos.fov==fov].values
        if readouts is None:
            self.readouts = dorg.get_data_info_df()['readout_name'].unique()
        else:
            self.readouts = readouts

    def get_nreadouts(self) -> int:
        """
        Get the number of readouts in the tile.
        """
        return len(self.readouts)

    def get_nrounds(self) -> int:
        """
        Get the number of z planes in the tile.
        """
        return len(self.dorg.get_data_info_df()['round'].unique())


    def get_zpos(self) -> np.ndarray:
        """
        Get the z positions for the tile.
        """
        return self.dorg.get_z_positions()

    def get_nz(self) -> int:
        """
        Get the number of z planes in the tile.
        """
        return len(self.get_zpos())

    def _load_data_from_annot(self, annot:pd.DataFrame):
        concat_data = []
        combined_annot = []
        all_z = []
        curr_z = 0
        for r in self.readouts:
            curr_annot = annot[annot['readout_name'] == r]
            image_type = curr_annot['image_type'].values[0]
            curr_round = curr_annot['round'].values[0]
            z_planes = curr_annot['frame_idx'].values
            curr_data = self.dataset.get_data_by_round_and_fov(image_type, curr_round, self.fov)
            # get the z planes corresponding to the channel
            curr_data = curr_data[z_planes]
            for j in range(curr_data.shape[0]):
                all_z.append(curr_z)
                curr_z += 1
            concat_data.append(curr_data)
            combined_annot.append(curr_annot)
        combined_annot = pd.concat(combined_annot)
        coords = {dim: (('i'), combined_annot[dim].values) for dim in combined_annot.columns}
        coords['i'] = np.array(all_z)
        coords['x'] = np.arange(curr_data.shape[1])
        coords['y'] = np.arange(curr_data.shape[2])

        # make combined data
        concat_data = da.concatenate(concat_data)

        # apply image transformations based on microscope if necessary
        if self.transpose or self.flip_vertical or self.flip_horizontal:
            z, x, y = concat_data.shape
            if concat_data.chunksize != (1, x, y):
                concat_data = concat_data.rechunk((1,x,y))
            if self.flip_vertical:
                #print("Flip vertical")
                concat_data = imagestack_flipud(concat_data, in_place=True)
            if self.flip_horizontal:
                #print("Flip horizontal")
                concat_data = imagestack_fliplr(concat_data, in_place=True)
            if self.transpose:
                #print("Transpose")
                concat_data = imagestack_transpose(concat_data, in_place=True)
    
        concat_data = xr.DataArray(concat_data, dims=('i','x','y'), coords=coords)

        #index_names_to_use = [i for i in index_names if i in annotation.columns]
        #concat_data = concat_data.set_index(i=index_names_to_use)
        concat_data.attrs['fov_id'] = self.fov
        if self.position is not None:
            concat_data.attrs['position'] = self.position

        return concat_data

    def get_flat_data(self) -> xr.DataArray:
        """
        Get the raw data for the tile. 
        """
        # iterate through the data organization and get the filenames for the given fov

        df = self.dorg.get_data_info_df()
        if self.readouts is not None:
            self.annotation = df[df['readout_name'].isin(self.readouts)]
        #df = df.sort_values(by=['readout_name', 'round'])
        return self._load_data_from_annot(df)

    def get_data(self) -> xr.DataArray:
        """
        Get the unrolled stack for the given fov, round, and channel.
        """
        data = self.get_flat_data()
        unrolled = unroll_tile(data, coords=['readout_name', 'z'], swap_dims=True, rechunk=False)
        _, _, n_x, n_y = unrolled.shape
        unrolled = unrolled.chunk((1,1, n_x, n_y))
        return unrolled

    def get_registered_data(self) -> xr.DataArray:
        """
        Get the registered stack for the given fov, round, and channel.
        """
        data = self.get_flat_data()
        fids = self.get_fiducials()
        data, shifts = register_tile_by_image_index(fids, data, reference_idx=self.ref_idx, mode='mirror', in_place=False) 
        unrolled = unroll_tile(data, coords=['readout_name', 'z'], swap_dims=True, rechunk=False)
        _, _, n_x, n_y = unrolled.shape
        unrolled = unrolled.chunk((1,1, n_x, n_y))
        return unrolled, shifts

    def get_fiducials(self) -> xr.DataArray:
        """
        Get the fiducials for the given fov.
        """
        fid_df = self.dorg.get_fiducial_info_df()
        fid_df = fid_df[fid_df['readout_name'].isin(self.readouts)]
        # get the subset of fiducials corresponding to the readouts
        return self._load_data_from_annot(fid_df)

    def get_registered_fiducials(self) -> xr.DataArray:
        """
        Get the registered fiducials for the given fov.
        """
        fids = self.get_fiducials()
        data, shifts = register_tile_by_image_index(fids, fids, reference_idx=self.ref_idx, mode='mirror', in_place=False)
        return data

# convenience functions to load tiles from the data organization
def load_tiles_from_dataorganization(data_org: DataOrganization, 
                                     dataset: ExperimentData,
                                     readouts:Optional[List[str]]=None,
                                     fov_subset:Optional[List]=None,
                                     flip_vertical:Optional[bool]=False,
                                     flip_horizontal:Optional[bool]=True,
                                     transpose:Optional[bool]=True)-> List[TileLoader]:
    """
    Load all the tiles from the dataorganization. Returns a list of tile objects .
    Can optionally specify number of FOVs and just subset of channels to load
    """
    if fov_subset is None:
        fovs = dataset.get_nfovs()
    else:
        fovs = fov_subset
    tiles = []
    for i in tqdm(range(fovs)):
        tiles.append(TileLoader(i, data_org, dataset, readouts=readouts, transpose=transpose, flip_vertical=flip_vertical, flip_horizontal=flip_horizontal))
    return tiles

# This function is deprecated now that loading the fiducials is pushed to the TileLoader class
def load_fiducial_tiles_from_dataorganization(data_org: DataOrganization, 
                                              fov_subset:Optional[List]=None,
                                              positions:Optional[np.ndarray]=None,
                                              flip_vertical:bool=False,
                                              flip_horizontal:bool=True,
                                              transpose:bool=True) -> List[xr.DataArray]:
    """
    Returns a list of tiles for the fiducial images.
    Each tile has an Nchannel x Nx x Ny ImageStack. 
    """
    if fov_subset is None:
        fovs = data_org.get_fovs()
    else:
        fovs = fov_subset
    tiles = []
    # this gives a dataframe with a row for each channel, but we really want it with a row for each image
    fiducial_info = data_org.get_fiducial_info_df(per_channel=False)

    for i in tqdm(fovs):
        if positions is not None:
            pos = positions[i,:]
        else:
            pos = None
        fnames = data_org.get_fiducial_filenames_for_fov(i, per_channel=False)
        tiles.append(load_tile_from_files(fnames, i, annotation=fiducial_info, position=pos, transpose=transpose, flip_vertical=flip_vertical, flip_horizontal=flip_horizontal))
    return tiles

def unroll_tile(data: xr.DataArray, coords=['name', 'z'], swap_dims=False, rechunk=True) -> xr.DataArray:
    """
    Reshape DataArray to have dimensions of Nchannel x Nz x Nx x Ny or Nz x Nchannel x Nx x Ny
    Arguments:
        data: DataArray with dimensions (Nz*Nchannel) x Nx x Ny
        coords: list of coords to use for unrolling. The first coord will be the one that is unrolled first, and will be used in its same order.
        swap_dims: if False, will swap the dimensions to be Nchannel x Nz x Nx x Ny. Otherwise, will be Nz x Nchannel x Nx x Ny
    """
    index_names = unique_unsorted(data.coords[coords[0]].values)

    full_coords = coords + ['x','y']
    unrolled = data.set_index(i=coords)
    unrolled = unrolled.unstack('i').transpose(*(full_coords))
    unrolled = xr.concat([unrolled.sel({coords[0]:n}) for n in index_names], dim=coords[0])
    if swap_dims:
        swapped_dims = [coords[1], coords[0], 'x', 'y']
        unrolled = unrolled.transpose(*(swapped_dims))
    if rechunk:
        unrolled = unrolled.chunk((1, unrolled.shape[1], unrolled.shape[2], unrolled.shape[3]))
    return unrolled

def roll_tile(data: xr.DataArray, coords=['name','z'])-> xr.DataArray:
    """
    Reshape DataArray to have dimensions of Nchannel x Nz x Nx x Ny 
    Assumes has i,x,y coords
    """
    temp = data.transpose(*(coords))
    return temp.stack(i=coords).transpose('i','x','y')

def subset_tile_by_image_index(data: xr.DataArray, image_idx: List[int]) -> xr.DataArray:
    """
    Subset the data to only the specified bits.
    """
    return data[data.coords['image_idx'].isin(image_idx)]

def subset_tile_by_channels(data: xr.DataArray, channels: List[int]) -> xr.DataArray:
    """
    Subset the data to only the specified bits.
    """
    return data[data.coords['name'].isin(channels)]

def subset_tile_by_readouts(data: xr.DataArray, readouts: List[int]) -> xr.DataArray:
    """
    Subset the data to only the specified bits.
    """
    return data[data.coords['readout_name'].isin(readouts)]


def concatenate_tiles(tiles: List[xr.DataArray], dim='i') -> xr.DataArray:
    """
    Concatenate a list of imagestacks into a single imagestack.
    """
    return xr.concat([i.xarray for i in tiles], dim=dim)

def get_tile_image_index(data: xr.DataArray) -> List[int]:
    return unique_unsorted(data.coords['image_idx'].values)

def get_tile_num_channels(data: xr.DataArray) -> int:
    return len(np.unique(data.coords['name'].values))

def get_tile_num_zplanes(data: xr.DataArray) -> int:
    return len(np.unique(data.coords['z'].values))

def get_tile_num_rounds(data: xr.DataArray) -> int:
    return len(np.unique(data.coords['round'].values))

def get_tile_shape(data: xr.DataArray) -> Tuple[int, int]:
    return (data.sizes['x'], data.sizes['y'])


def cast_imagestack_dtype(data: xr.DataArray, dtype:np.dtype, in_place=False) -> xr.DataArray:    
    if not in_place:
        temp = data.copy()
        temp = cast_imagestack_dtype(temp,dtype, in_place=True)
        return temp
    else:
        return data.astype(dtype)

#
# BELOW ARE OLDER FUNCTIONS FOR TESTING
#
def load_tile_from_files(fnames:List[str], fov_id: int, position: np.ndarray=None, annotation: pd.DataFrame=None,
                         transpose:bool=False, flip_vertical:bool=False, flip_horizontal:bool=False) -> xr.DataArray:
                    #index_names=['name','z','color','image_idx','readout_name','round', 'frame_idx']) -> xr.DataArray:
    """
    This is an old test function to directly load a tile from disk.
    Concatenates the images from reach round.
    """
    data = [imread.imread(i) for i in fnames]

    # subset the data to the channels that are in the annotation
    all_z = []
    curr_z = 0

    dims = ('i','x','y')
    if annotation is not None:
        reordered_annot = []
        for i in range(len(data)):
            # handle loading multiple things from the same image_idx
            curr_annot = annotation[annotation['image_idx'] == i]
            curr_frames = np.array(curr_annot['frame_idx'].values)
            data[i] = data[i][curr_frames, :, :]
            for j in range(len(curr_frames)):
                all_z.append(curr_z)
                curr_z += 1
            reordered_annot.append(curr_annot)
        annotation = pd.concat(reordered_annot)
        coords = {dim: (('i'), annotation[dim].values) for dim in annotation.columns}
        coords['i'] = np.array(all_z)
    else:
        coords = {}
        coords['i'] = np.arange(np.sum([i.shape[0] for i in data]))
    coords['x'] = np.arange(data[0].shape[1])
    coords['y'] = np.arange(data[0].shape[2])

    # make combined data
    concat_data = da.concatenate(data)

    # apply image transformations based on microscope if necessary
    if transpose or flip_vertical or flip_horizontal:
        z, x, y = concat_data.shape
        if concat_data.chunksize != (1, x, y):
            concat_data = concat_data.rechunk((1,x,y))
        if flip_vertical:
            #print("Flip vertical")
            concat_data = imagestack_flipud(concat_data, in_place=True)
        if flip_horizontal:
            #print("Flip horizontal")
            concat_data = imagestack_fliplr(concat_data, in_place=True)
        if transpose:
            #print("Transpose")
            concat_data = imagestack_transpose(concat_data, in_place=True)
 
    concat_data = xr.DataArray(concat_data, dims=dims, coords=coords)

    #index_names_to_use = [i for i in index_names if i in annotation.columns]
    #concat_data = concat_data.set_index(i=index_names_to_use)
    concat_data.attrs['fov_id'] = fov_id
    if position is not None:
        concat_data.attrs['position'] = position

    return concat_data

