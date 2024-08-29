from typing import Optional, List, Tuple
import xarray as xr
import dask.distributed as dd
import numpy as np
import pandas as pd
import os
from pftools.pipeline.util.spotutils import find_seeds_across_rounds
from pftools.pipeline.processing.globalalign import SimpleGlobalAlignment
from pftools.pipeline.core.algorithm import ParallelAnalysisTask

class QuantifySMFish(ParallelAnalysisTask):
    """
    Class to quantify smFISH data using spot finding. 
    """
    def __init__(self, client:dd.Client, tiles:List[xr.DataArray],
                 output_path:str, global_align:Optional[SimpleGlobalAlignment]=None):
        super().__init__(client, output_path)
        self.global_align = global_align
        self.tiles = tiles

    def find_spots_for_tile(self, tile_idx:int, tile: xr.DataArray, filt_size:int=3, gfilt_size:int=5, th_seed:float=3.) -> np.ndarray:
        """
        Find spots for a single tile across all channels.
        """
        n_z, n_chan, n_x, n_y = tile.shape
        chan_names = tile.coords['channel'].values
        spots = find_seeds_across_rounds(tile.compute(), filt_size, gfilt_size, th_seed)
        chans = [chan_names[i] for i in spots[:,0]]
        spots_df = pd.DataFrame(spots, columns=['round','z', 'x', 'y', 'h'])
        spots_df['global_z'] = spots_df['z']
        spots_df['fov'] = tile_idx
        spots_df['channel'] = chans
        return spots_df

    def run(self, filt_size:int=2, gfilt_size:int=5, th_seed:float=3.):
        # find spots
        self.logger.info("Finding spots...")
        futures = [self.client.submit(self.find_spots_for_tile, tile, filt_size, gfilt_size, th_seed) for tile in self.tiles]
        spots = pd.concat(self.client.gather(futures))
        self.logger.info("Done finding spots")
        
        # return results with tile index added
        if self.global_align is not None:
            for i in spots['fov'].unique():
                idx = spots['fov']==i
                spots[idx][['global_x', 'global_y']] = self.global_align.fov_coordinate_array_to_global(i, spots[idx][['x', 'y']].values)

        # save results
        self.logger.info("Saving results...")
        if not os.path.exists(self._output_path):
            os.makedirs(self._output_path)
        spots.to_csv(os.path.join(self._output_path, 'spots.csv'))

        return spots