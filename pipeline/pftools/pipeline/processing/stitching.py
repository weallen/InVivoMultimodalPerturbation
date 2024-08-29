import dask.array as da
import dask.distributed as dd
import numpy as np
import xarray as xr
from typing import List

from pftools.pipeline.processing.globalalign import SimpleGlobalAlignment

class Stitcher:
    def __init__(self, client:dd.Client, output_path:str, global_align:SimpleGlobalAlignment):
        self.client = client
        self.global_align = global_align
        self.output_path = output_path

    def run(self, tiles:List[xr.DataArray]):
        """
        Stitch together the tiles into a single image.
        """

def register_overlapping_tiles(tile1:da.Array, tile2:da.Array) -> Tuple[da.Array,da.Array]:
    """
    Register two overlapping tiles.
    """
    pass

def compute_optimal_overlap():
    pass

def find_overlapping_tiles(global_align:SimpleGlobalAlignment) -> List[Tuple[int,int]]:
    fov_boxes = global_align.get_fov_boxes()
    overlapping_tiles = []
    n_fovs = len(fov_boxes)
    for i in range(n_fovs):
        box = fov_boxes[i]
        for j in range(n_fovs):
            other_box = fov_boxes[j]
            if box.intersects(other_box) and i!=j:
                overlapping_tiles.append((i,j))
    return overlapping_tiles