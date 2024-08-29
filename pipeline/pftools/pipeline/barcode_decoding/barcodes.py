import numpy as np
from typing import List, Tuple, Dict, Optional
import dask.array as da
import dask.delayed as dd
from pftools.pipeline.core import Codebook

class BarcodeTile(object):
    """
    Object that holds all the information about a single tile across rounds and channels,
    but lazily loads the data.
    """
    def __init__(self, common_bit_path:str, barcode_paths:str, tile_id:int):
        """
        Initialize the tile object
        """
        self._common_bit_path = common_bit_path
        self._barcode_paths = barcode_paths
        self._tile_id = tile_id

    def get_tile_id(self) -> int:
        """
        Get the tile id
        """
        return self._tile_id
    
    def get_fov_path(self) -> str:
        """
        Get the path to the fov
        """
        return self._fov_path  

    def get_num_rounds(self) -> int:
        """
        Get the number of rounds
        """
        pass

    def get_num_channels(self) -> int:
        """
        Get the number of channels
        """
        pass

    def get_num_z_planes(self) -> int:
        """
        Get the number of z planes
        """
        pass

    def get_num_bits(self) -> int:
        """
        Get the number of bits
        """
        pass    

class BarcodeDecoder(object):
    """
    Decode barcodes from images for a single FOV.
    Assumes that has enough memory to load all images for a single FOV.
    This will load the images for a single FOV, register them, and then decode the barcodes.
    """

    def __init__(self, fov:BarcodeTile, codebook:Codebook, output_path:str):
        """
        Initialize the barcode decoder
        """
        self._fov = fov
        self._codebook = codebook
        self._output_path = output_path

    def decode(self):
        """
        Decode the barcodes
        """
        self._identify_common_bit()
    
    def _identify_common_bit(self):
        pass

    def _load_fov(self) -> np.ndarray:
        pass

    def _register_images(self) -> np.ndarray:
        pass

    def _decode_fov(self) -> np.ndarray:
        pass

    def _save_decoded(self) -> np.ndarray:
        pass

