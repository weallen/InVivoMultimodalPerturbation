import logging
import os
import pandas as pd
import numpy as np
import dask.array as da
import dask.delayed as dd
import sys
import xarray as xr
from typing import Tuple, List, Optional
from pftools.pipeline.core.microscope import MicroscopeInformation
from pftools.pipeline.core.codebook import Codebook
from pftools.pipeline.core.dataorganization import DataOrganization
from pftools.pipeline.core.tile import load_tiles_from_dataorganization
from pftools.pipeline.processing.globalalign import SimpleGlobalAlignment
from pftools.pipeline.core.dataset import ExperimentDataFS, ExperimentDataZarr


def convert_db_to_filemap(db: pd.DataFrame):
    """
    Convert a database to a filemap.
    FileMap files are: imageType,fov,imagingRound,imagePath
    where imagePath is the path to the image relative to the root of the data directory.
    """
    def _truncate_file_path(path) -> None:
        head, tail = os.path.split(path)
        return tail
    
    if "filename" not in db.columns:
        db['filename'] = db['prefix'] + "_" + db['fov'].astype(str) + "_" + db['round'].astype(str) + ".tif"

    return pd.DataFrame({
        'imageType': db['prefix'],
        'fov': db['fov'],
        'imagingRound': db['round'],
        'imagePath': db['filename'].apply(lambda x: _truncate_file_path(x))
    })


class BaseExperiment:
    """
    Lightweight base class for specifying everything about an experiment. 
    Loads the microscope information, codebook, and tile information, but not actual images.
    This is used by the pipeline to specify the experiment, and then the pipeline will load the images and process them.
    """

    def __init__(self, data_path:str, 
                 output_path: str,
                 dataorganization_path:str, 
                 microscope_path:Optional[str]=None, 
                 codebook_path:Optional[str]=None,
                 max_fov:Optional[int]=None,
                 force_rebuild:Optional[bool]=False):
                #positions_path:Optional[str]=None):

        self.data_path = data_path
        self.output_path = output_path
        self.max_fov = max_fov
        if not os.path.exists(data_path):
            raise ValueError(f"Data path {data_path} does not exist.")

        if "zarr" in data_path:
            if data_path[-1] != '/':
                data_path = data_path + '/' # make sure ends in slash so recognizes as directory
            self.data = ExperimentDataZarr(data_path, force_rebuild=force_rebuild)
        else:
            self.data = ExperimentDataFS(data_path, force_rebuild=force_rebuild)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        #self.logger = self._create_logger()

        #filemap_path = os.path.join(self.output_path, 'filemap.csv')
       # if os.path.exists(filemap_path):
        #fileMap = convert_db_to_filemap(self._data.db)

        self.data_organization = DataOrganization(dataorganization_path, data_path)
        #else: 
        #    self.data_organization = DataOrganization(dataorganization_path, data_path, max_fov=max_fov)
            # cache the filemap information
            #self.data_organization.save_filemap_df(os.path.join(self.output_path, 'filemap.csv'))

        if codebook_path is not None:
            # not all experiments have codebooks
            self.codebook = Codebook(codebook_path)
            self.codebook.save_codebook_df(os.path.join(self.output_path, 'codebook.csv'))
        
        if microscope_path is not None:
            self.microscope_information = MicroscopeInformation(microscope_path)
            self.microscope_information.save_microscope_yaml(os.path.join(self.output_path, 'microscope.yaml'))
        else:
            self.microscope_information = MicroscopeInformation()
        # load positions
        self.positions = self.data.get_positions()#pd.read_csv(positions_path, header=None) 
        # check if first row is the same as second row, if so, remove it
        if self.positions.iloc[0].x == self.positions.iloc[1].x and self.positions.iloc[0].y == self.positions.iloc[1].y:
            print("Removing duplicate first row")
            self.positions = self.positions.iloc[1:,:]
        #self.positions.columns = ['x', 'y']
        self.positions['fov'] = np.arange(self.positions.shape[0])
        if max_fov is not None:
            self.positions = self.positions.iloc[:max_fov,:]
        self.positions.to_csv(os.path.join(self.output_path, 'positions.csv'), index=False, header=True)

        # get global alignment
        z_pos = self.data_organization.get_z_positions()
        pos = self.positions.loc[:,["x","y"]].values
        self.global_align = SimpleGlobalAlignment(self.output_path, pos, z_pos,
                                                  micron_per_pixel=float(self.microscope_information.microns_per_pixel),
                                                  image_dimensions=self.microscope_information.image_dimensions)

    #def get_reference_images_by_readout(self, readout_name:str) -> xr.DataArray:
    #    """
    #    Get the reference images for a particular readout that everything will be aligned to. 
    #    """
    #    # get the image index for a particular readout
    #    temp = subset_tile_by_readouts(self.tiles[0], [readout_name])
    #    image_idx = temp.coords['image_idx'].values[0]
    #    # get the reference image
    #    return [subset_tile_by_image_index(t, [image_idx])[0] for t in self.fiducial_tiles]

    #def get_tile_subset_for_readouts(self, readout_names:List[str]) -> Tuple[List[xr.DataArray], List[xr.DataArray]]:
    #    # get subset of tils for readout names
    #    tiles = [subset_tile_by_readouts(t, readout_names) for t in self.tiles]
    #    # get image_idx for fiducial tiles, using first tile 
    #    image_idx = tiles[0].coords['image_idx'] 
    #    # get subset of fiducial tiles for readout names
    #    fid_tiles = [subset_tile_by_image_index(t, image_idx) for t in self.fiducial_tiles]
    #    return tiles, fid_tiles

    def load_data(self, n_fov:Optional[int]=None, readout_names:Optional[List[str]]=None):
        """
        Lazily load the data specified in the dataorganization.
        """
        transpose = self.microscope_information.transpose
        flip_vertical = self.microscope_information.flip_vertical
        flip_horizontal = self.microscope_information.flip_horizontal
        if n_fov is None:
            n_fov = self.positions.shape[0]

        self.tiles = load_tiles_from_dataorganization(self.data_organization, self.data, readout_names, n_fov, flip_vertical, flip_horizontal, transpose)
        #self.fiducial_tiles = load_fiducial_tiles_from_dataorganization(self.data_organization, range(n_fov), flip_vertical, flip_horizontal, transpose)
