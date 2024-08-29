import numpy as np
import cv2
from pftools.pipeline.core.algorithm import AnalysisTask
import xarray as xr
from typing import List

class GenerateMosaic(AnalysisTask):
    def __init__(self, output_path:str,
                 tiles:List[xr.DataArray],
                 fov_crop_width:int = 0,
                 microns_per_pixel:float = 0.108,
                 tile_size:int = 2048,
                 fov_positions:int = np.ndarray,
                 draw_fov_labels:bool = False):
        super().__init__(output_path)
        self.tile_size = tile_size
        self.overlap = overlap

   # def run(self) -> None:
   #     # get the list of tiles
   #     tile_list = self._get_tile_list(input_path)


        # create the mosaic
   #     mosaic = self._create_mosaic(mosaic_dims)

   #def _create_mosaic(self, mosaic_dims):
   #    pass

