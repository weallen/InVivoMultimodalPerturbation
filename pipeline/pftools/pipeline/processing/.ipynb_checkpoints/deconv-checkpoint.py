
import dask.distributed as dd

import dask.distributed as dd
from pftools.pipeline.core.algorithm import ParallelTileAnalysisTask
from pftools.pipeline.util.dw import run_dw, run_dw_tiled

class Deconvolve(ParallelTileAnalysisTask):
    def __init__(self):
        pass 

    def _process_tile(self, tile_idx: int) -> None:
        return super()._process_tile(tile_idx)
    
    def _tile_processing_done_callback(self, tile_idx: int, future: dd.Future) -> None:
        return super()._tile_processing_done_callback(tile_idx, future)