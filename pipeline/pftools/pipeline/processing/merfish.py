from typing import List, Tuple, Optional, Union
import dask.array as da
from dask.delayed import delayed
import dask.distributed as dd
import xarray as xr
import numpy as np
import os
import logging
from dataclasses import dataclass
#from multiprocessing import Process
import zarr
import pandas as pd
from tqdm import tqdm

from pftools.pipeline.core.tile import TileLoader
from pftools.pipeline.core.codebook import Codebook
from pftools.pipeline.processing.optimization import select_tiles_for_optimization, optimize_scalefactors_across_tiles
from pftools.pipeline.util.preprocessing import imagestack_lowpass_filter, imagestack_deconvolve, imagestack_highpass_filter
from pftools.pipeline.processing.decoding import calculate_normalized_barcodes, imagestack_decode_pixels_dask, decode_pixels_singleplane_fast
from pftools.pipeline.core.algorithm import ParallelTileAnalysisTask
from pftools.pipeline.processing.globalalign import SimpleGlobalAlignment
from pftools.pipeline.util.moleculeutil import extract_molecules_from_decoded_image, remove_zplane_duplicates_all_barcodeids, crop_molecules, molecule_count_histograms, blank_fraction_histogram, calculate_threshold_for_misidentification_rate, assign_gene_symbols_to_mols
from pftools.pipeline.util.barcode_classifier import predict_molecule_quality, calculate_threshold_for_misidentification_rate_ml

from pftools.pipeline.processing.registration import register_tile_to_reference

@dataclass
class PixelBasedDecodeParams:
    """
    A dataclass for storing the decoding parameters.
    """
    low_pass_sigma:float=1
    decon_sigma:int=2
    decon_filter_size:Optional[int]=9
    decon_iterations:int=10
    high_pass_sigma:int=3
    extract_backgrounds:bool=True
    distance_threshold:float=0.8
    magnitude_threshold:float=0.2
    crop_fraction:float = 0 #previously 0.05, don't crop at all
    min_area_size:int = 4
    remove_z_duplicated_molecules:bool=True
    z_duplicate_zplane_threshold:int=1
    z_duplicate_xy_pixel_threshold:float=np.sqrt(2)
    keep_blank_barcodes:bool=True
    area_threshold:int=4
    misidentification_rate:float=0.05
    do_deconv:bool=False
    force_rerun:bool=False
    #save_normalized_pixel_traces:bool=False
    #compute_reg_fiducials:bool=False
    #save_molecules:bool=True

class PixelBasedDecoder(ParallelTileAnalysisTask):
    """
    Takes in MERFISH data and decodes it.
    """
    def __init__(self, client:dd.Client,
                 tiles:List[TileLoader],
                 codebook: Codebook,
                 params:PixelBasedDecodeParams,
                 output_base_path:str,
                 global_align:Optional[SimpleGlobalAlignment]=None):
        # initialize superclass
        super().__init__(client, tiles, output_base_path)

        self.params = params
        self.codebook = codebook
        self.decoding_matrix = calculate_normalized_barcodes(self.codebook)
        self.background = np.zeros(self.decoding_matrix.shape[1])
        self.scale_factor = np.ones(self.decoding_matrix.shape[1])
        self.global_align = global_align
        # Initialize stuff for optimization to None
        self.barcode_counts = None
        self.previous_scale_factors = None
        self.previous_backgrounds = None
        self.scale_refactors = None
        self.background_refactors = None

    def optimize(self, n_fov=50, n_iter=10) -> None:
        """
        Initialize the decoder by optimizing the scale_factors, backgrounds, and chromatic offsets.
        """
        self.logger.info("Selecting and registering tiles for optimization")
        reg_tiles = select_tiles_for_optimization(self.client, self.tiles, n_fov)
        self.logger.info("Optimizing scale factors across tiles")
        self.barcode_counts, self.previous_scale_factors, \
            self.previous_backgrounds, self.scale_refactors, self.background_refactors \
                = optimize_scalefactors_across_tiles(self.client, reg_tiles, self.codebook,
                                                     n_iter=n_iter,
                                                     distance_threshold=self.params.distance_threshold,
                                                     magnitude_threshold=self.params.magnitude_threshold,
                                                     low_pass_sigma=self.params.low_pass_sigma,
                                                     extract_backgrounds=self.params.extract_backgrounds,
                                                     logger=self.logger)
        self.scale_factor = self.previous_scale_factors[-1]
        self.background = self.previous_backgrounds[-1]

        self._save_optimization_results()
        return

    def _save_optimization_results(self) -> None:
        """
        Save out the results of the optimization.
        """
        # save out the results of the optimization
        optimization_path = os.path.join(self.output_path, 'optimization')
        if not os.path.exists(optimization_path):
            os.makedirs(optimization_path)
        np.save(os.path.join(optimization_path, 'barcode_counts.npy'), self.barcode_counts)
        np.save(os.path.join(optimization_path, 'previous_scale_factors.npy'), self.previous_scale_factors)
        np.save(os.path.join(optimization_path, 'previous_backgrounds.npy'), self.previous_backgrounds)
        np.save(os.path.join(optimization_path, 'scale_refactors.npy'), self.scale_refactors)
        np.save(os.path.join(optimization_path, 'background_refactors.npy'), self.background_refactors)
        return


    def load_scale_factors(self) -> None:
        """
        Load in the scale factors from a previous optimization.
        """
        self.logger.info("Loading scale factors from previous optimization")
        optimization_path = os.path.join(self.output_path, 'optimization')
        if os.path.exists(os.path.join(optimization_path, 'barcode_counts.npy')):
            self.barcode_counts = np.load(os.path.join(optimization_path, 'barcode_counts.npy'))
            self.previous_scale_factors = np.load(os.path.join(optimization_path, 'previous_scale_factors.npy'))
            self.previous_backgrounds = np.load(os.path.join(optimization_path, 'previous_backgrounds.npy'))
            self.scale_refactors = np.load(os.path.join(optimization_path, 'scale_refactors.npy'))
            self.background_refactors = np.load(os.path.join(optimization_path, 'background_refactors.npy'))
            self.scale_factor = self.previous_scale_factors[-1]
            self.background = self.previous_backgrounds[-1]
            return True
        else:
            self.logger.info("No previous optimization found")
            return False

    def _process_tile(self, tile_idx: int) -> Tuple[int, dd.Future]:
        # check that tile hasn't already been processed
        tile_path = os.path.join(self.output_path, 'decoding', f'tile_{tile_idx}', 'molecules.parquet')
        if os.path.exists(tile_path) and not self.params.force_rerun:
            self.logger.info(f"Tile {tile_idx} already processed, skipping")
            return tile_idx, None
        else:
            #if self.ref_imgs is None:
                # get the reference image for this tile
            #    ref_img = self.fiducial_tiles[tile_idx]
            #    ref_img = ref_img[ref_img.coords['image_idx'] == ref_img.coords['image_idx'].values.min()]
            #else:
            #    ref_img = self.ref_imgs[tile_idx]
            return tile_idx, self.client.submit(decode_tile, tile_idx, self.tiles[tile_idx],
                                                self.params,
                                                self.global_align,
                                                self.decoding_matrix, self.scale_factor, self.background)

    def _tile_processing_done_callback(self, tile_idx: int, future: dd.Future) -> None:
        #shifts, decoded_image, pixel_magnitudes, normalized_pixel_traces, dist, reg_fiducials = future.result() # type: ignore
        if future is not None:
            shifts, mols, decoded_image, pixel_magnitudes, dist = future.result()
            self._save_decoding_for_tile(tile_idx, mols, shifts, decoded_image, pixel_magnitudes, dist) # type: ignore
            del decoded_image, pixel_magnitudes, dist
            return
        else:
            return

    def _save_decoding_for_tile(self, tile_idx:int, molecules:pd.DataFrame, shifts:np.ndarray, decoded_image:np.ndarray,
                                pixel_magnitudes:np.ndarray, dist:np.ndarray) -> None:

                        #self.logger.info("Filtering molecules based on min area and magnitude for tile {tile_idx}")
            #molecules = filter_molecules(molecules, self.codebook, self.params.keep_blank_barcodes, self.params.min_area_size)
            #molecules = molecules[(molecules['area'] > self.params.area_threshold) & \
            #                    (molecules['mean_distance'] <= self.params.distance_threshold) & \
            #                    (molecules['mean_intensity'] >= self.params.magnitude_threshold)]

        # create the output directory if it doesn't exist
        self.logger.info(f"Saving decoding for tile {tile_idx}")
        tile_path = os.path.join(self.output_path, 'decoding', f'tile_{tile_idx}')
        if not os.path.exists(os.path.join(self.output_path, tile_path)):
            os.makedirs(os.path.join(self.output_path, tile_path))

        # save out the molecules
        self._save_molecules_for_tile(tile_path, molecules)

        # save out the decoded image
        # skip this for now
        #self._save_images_for_tile(tile_path, decoded_image, pixel_magnitudes, dist)

        # save out the shifts
        np.save(os.path.join(tile_path, 'shifts.npy'), shifts)
        # don't save this -- too big!
        #if self.params.save_normalized_pixel_traces:
        #    zarr.save(os.path.join(tile_path, 'normalized_pixel_traces.zarr'), normalized_pixel_traces)
        #if registered_fiducials is not None:
        #    zarr.save(os.path.join(tile_path, 'registered_fiducials.zarr'), registered_fiducials)
        return

    def _save_images_for_tile(self, tile_path:str, decoded_image:np.ndarray, pixel_magnitudes:np.ndarray, dist:np.ndarray) -> None:
        zarr.save(os.path.join(tile_path, 'decoded_image.zarr'), decoded_image)
        zarr.save(os.path.join(tile_path, 'pixel_magnitudes.zarr'), pixel_magnitudes)
        zarr.save(os.path.join(tile_path, 'dists.zarr'), dist)
        return

    def _save_molecules_for_tile(self, tile_path:str, molecules:pd.DataFrame) -> None:
        # save out the results
        if self.params.remove_z_duplicated_molecules:
            z_pos = list(np.unique(molecules['z']))
            molecules = remove_zplane_duplicates_all_barcodeids(molecules, self.params.z_duplicate_zplane_threshold,
                                                    self.params.z_duplicate_xy_pixel_threshold, z_pos)
        molecules = molecules[molecules.barcode_id != -1]
        molecules = molecules[molecules.barcode_id != 65535]
        molecules.to_parquet(os.path.join(tile_path, f'molecules.parquet'))


    def _load_all_molecules(self) -> pd.DataFrame:
        """
        Load all molecules from the decoding.
        """
        self.logger.info("Loading molecules")
        molecules = []
        for tile_idx in tqdm(range(len(self.tiles))):
            tile_path = os.path.join(self.output_path, 'decoding', f'tile_{tile_idx}')
            molecules.append(pd.read_parquet(os.path.join(tile_path, 'molecules.parquet')))
        molecules = pd.concat(molecules, ignore_index=True)
        molecules = molecules[molecules.barcode_id != -1]
        molecules = molecules[molecules.barcode_id != 65535]
        return molecules

    def _postrun_processing(self) -> None:
        # TODO implement this
        molecules = self._load_all_molecules()
        molecules = assign_gene_symbols_to_mols(molecules, self.codebook)
        molecules.to_parquet(os.path.join(self.output_path, 'molecules.parquet'), index=False)

        filtered_molecules = self._dynamic_filter_molecules(molecules)
        self.logger.info(f"Filtered down to {len(filtered_molecules)} molecules from {len(molecules)} ({len(filtered_molecules)/len(molecules)*100:0.02f}%)")
        #nx, ny = self.tiles[0].get_flat_data().shape[1:]
        self.logger.info("Saving filtered molecules")
        filtered_molecules.to_parquet(os.path.join(self.output_path, 'filtered_molecules.parquet'), index=False)
        self.logger.info(f"Cropping down to {(1-self.params.crop_fraction)*100}% ({(1-self.params.crop_fraction)*nx} x {(1-self.params.crop_fraction)*ny}) of FOV")
        filtered_molecules = crop_molecules(filtered_molecules, self.params.crop_fraction, nx, ny)
        filtered_molecules.to_parquet(os.path.join(self.output_path, 'cropped_molecules.parquet'), index=False)
        self.logger.info("Done saving filtered molecules")

    def _dynamic_filter_molecules(self, molecules:pd.DataFrame) -> pd.DataFrame:
        """
        Filter barcodes based on dynamic thresholding.
        """
        self.logger.info("Filtering molecules")
        self.logger.info(f"Calculating threshold for misidentification rate {self.params.misidentification_rate}")
        n_coding = len(self.codebook.get_coding_indexes())
        n_blank = len(self.codebook.get_blank_indexes())
        coding_counts, blank_counts, intensity_bins, distance_bins, area_bins = molecule_count_histograms(molecules, self.codebook, self.params.distance_threshold)
        blank_frac = blank_fraction_histogram(coding_counts, blank_counts, n_coding, n_blank)
        threshold = calculate_threshold_for_misidentification_rate(self.params.misidentification_rate, coding_counts, blank_counts, blank_frac, n_coding, n_blank)
        self.logger.info(f"Identified threshold of {threshold}")
        select_data = molecules[["mean_intensity", "min_distance", 'area']].values
        select_data[:,0] = np.log10(select_data[:,0])
        barcode_bins = np.array(
            (np.digitize(select_data[:,0], intensity_bins, right=True),
            np.digitize(select_data[:,1], distance_bins, right=True),
            np.digitize(select_data[:,2], area_bins)))-1
        barcode_bins[0,:] = np.clip(barcode_bins[0,:], 0, blank_frac.shape[0]-1)
        barcode_bins[1,:] = np.clip(barcode_bins[1,:], 0, blank_frac.shape[1]-1)
        barcode_bins[2,:] = np.clip(barcode_bins[2,:], 0, blank_frac.shape[2]-1)
        raveled_indexes = np.ravel_multi_index(barcode_bins[:, :], blank_frac.shape)
        thresholded_blank_frac = blank_frac < threshold
        return molecules[np.take(thresholded_blank_frac, raveled_indexes)]

class RCAPixelBasedDecoder(PixelBasedDecoder):
    """
    Pixel based decoder for RCA amplicons.
    Optimizer uses spot-finding to find the amplicons and then decodes to initialize optimization of scale factors.
    Then does pixel based decoding
    Don't do LR deconv then blurring
    Don't do L2 norm during decoding.
    """
    def __init__(self, client:dd.Client,
                 tiles:List[xr.DataArray],
                 codebook: Codebook,
                 params:PixelBasedDecodeParams,
                 output_base_path:str,
                 global_align:Optional[SimpleGlobalAlignment]=None):
        # initialize superclass
        super().__init__(client, tiles, codebook, params, output_base_path, global_align)
        self.logger = logging.getLogger("RCAPixelBasedDecoder")

    def optimize(self, n_fov:int=50, n_iter:int=1, dist_thresh:float=0.8, mag_thresh:float=0.25, area_thresh:int=6):
        """
        Slightly tweaked optimization to just do high-pass filter and set background and foreground quantiles much higher.
        """
        self.logger.info("Selecting and registering tiles for optimization")
        reg_tiles = select_tiles_for_optimization(self.client, self.tiles, n_fov)
        self.logger.info("Optimizing scale factors across tiles")
        self.barcode_counts, self.previous_scale_factors, \
            self.previous_backgrounds, self.scale_refactors, self.background_refactors \
                = optimize_scalefactors_across_tiles(self.client, reg_tiles, self.codebook,
                                                     n_iter=n_iter,
                                                     distance_threshold=dist_thresh,
                                                     magnitude_threshold=mag_thresh,
                                                     area_threshold=area_thresh,
                                                     low_pass_sigma=self.params.low_pass_sigma,
                                                     extract_backgrounds=self.params.extract_backgrounds,
                                                     logger=self.logger,
                                                     do_deconv=self.params.do_deconv, # don't do deconvolution
                                                     background_quantile=0.001, # assumes that the amplicons are very bright
                                                     foreground_quantile=0.999,
                                                     use_ml_filt=True)
        self.scale_factor = self.previous_scale_factors[-1]
        self.background = self.previous_backgrounds[-1]

        self._save_optimization_results()
        return

    def _ml_filter_molecules(self, molecules:pd.DataFrame, misid_rate:float=0.5) -> pd.DataFrame:
        """
        Filter barcodes based on machine learning.
        """
        self.logger.info("Filtering molecules")
        self.logger.info(f"Calculating threshold for misidentification rate {self.params.misidentification_rate}")
        misid_rate = calculate_threshold_for_misidentification_rate_ml(molecules, self.codebook, target_rate=0.05)
        self.logger.info(f"Identified threshold of {misid_rate}")
        return molecules[molecules.quality < misid_rate]

    def _postrun_processing(self) -> None:
        # TODO implement this
        molecules = self._load_all_molecules()
        
        self.logger.info("Saving molecules")
        molecules = molecules[molecules.barcode_id != -1]
        #molecules = assign_gene_symbols_to_mols(molecules, self.codebook)
        molecules.to_parquet(os.path.join(self.output_path, 'molecules.parquet'), index=False)
        n_orig = molecules.shape[0]
        # do ML filtering
        self.logger.info("Predicting molecule quality")
        molecules = predict_molecule_quality(molecules, self.codebook)
        molecules = self._ml_filter_molecules(molecules)
        self.logger.info("Saving filtered molecules")
        molecules.to_parquet(os.path.join(self.output_path, 'filtered_molecules.parquet'), index=False)
        self.logger.info(f"Filtered down to {len(molecules)} molecules from {n_orig} ({len(molecules)/n_orig*100:0.02f}%)")
        #nx, ny = self.tiles[0].get_flat_data().shape[1:]
        #self.logger.info(f"Cropping down to {(1-self.params.crop_fraction)*100}% ({(1-self.params.crop_fraction)*nx} x {(1-self.params.crop_fraction)*ny}) of FOV")
        #filtered_molecules = crop_molecules(filtered_molecules, self.params.crop_fraction, nx, ny)
        #filtered_molecules.to_parquet(os.path.join(self.output_path, 'cropped_molecules.parquet'), index=False)

def extract_molecules_from_tile(tile_idx:int, decoded:np.ndarray, distances:np.ndarray,
                                    traces:np.ndarray, magnitudes:np.ndarray,
                                    params:PixelBasedDecodeParams,
                                    global_align:SimpleGlobalAlignment,
                                    bc_per_partition:int=50) -> pd.DataFrame:

    # partition barocodes into smaller number of chunks to avoid creating too many tasks
    # this is necessary because the number of barcodes can be very large
    curr_barcodes = np.unique(decoded)
    #decoded, distances, traces, magnitudes = self.client.scatter([decoded, distances, traces, magnitudes]) # type: ignore
    curr_barcodes = curr_barcodes[curr_barcodes != -1] # remove background/unassigned
    n_barcodes = curr_barcodes.shape[0]
    n_partitions = int(np.ceil(n_barcodes/bc_per_partition))
    curr_barcodes = np.array_split(curr_barcodes, n_partitions)
    #futures = [self.client.submit(extract_molecules_from_zplanes, tile_idx, decoded, distances, traces,
    #                                                            magnitudes, curr_barcodes[i],
    #                                                            self.params.crop_width,
    #                                                            self.params.min_area_size,
    #                                                            self.global_align) for i in range(n_partitions)]
    #results = self.client.gather(futures) # type: ignore
    crop_width = params.crop_fraction * decoded.shape[1]
    results = da.compute([delayed(extract_molecules_from_zplanes)(tile_idx, decoded, distances, traces,
                                                                    magnitudes, curr_barcodes[i],
                                                                    crop_width,
                                                                    params.min_area_size,
                                                                    global_align) for i in range(n_partitions)])[0]
    df = pd.concat(results, ignore_index=True) # type: ignore
    return df

def extract_molecules_from_zplanes(tile_idx:int, decoded:da.Array, distances:da.Array, traces:da.Array,
                                   magnitudes:da.Array, barcodes_indexes:np.ndarray,
                                   crop_width:int=100, min_area_size:int=3, global_align:Optional[SimpleGlobalAlignment]=None) -> pd.DataFrame:
    # use delayed functions for each z plane
    n_z = decoded.shape[0]
    z_pos = np.arange(n_z)
    zplanes = [extract_molecules_from_decoded_image(decoded[i], magnitudes[i], traces[i], distances[i], tile_idx, crop_width, i, min_area_size, barcodes_indexes) for i in range(n_z)]
    #zplanes = [delayed(extract_molecules_from_decoded_image)(da.squeeze(decoded[i]),
    #                                                            da.squeeze(magnitudes[i]),
    #                                                            da.squeeze(traces[i]),
    #                                                            da.squeeze(distances[i]),
    #                                                            tile_idx, crop_width, i,
    #                                                            min_area_size, barcodes_indexes) for i in range(n_z)]
    #zplanes = da.compute(zplanes)[0] # type: ignore
    for i in range(n_z):
        zplanes[i]['z'] = z_pos[i]#self.z_pos[i] # type: ignore
    zplanes = pd.concat(zplanes, ignore_index=True)  # type:ignore
    zplanes['fov'] = tile_idx
    if global_align is not None:
        centroids = np.array(zplanes[["z","x","y"]].values)
        global_centroids = global_align.fov_coordinate_array_to_global(tile_idx, centroids)
        zplanes['global_z'] = global_centroids[:,0]
        zplanes['global_x'] = global_centroids[:,1]
        zplanes['global_y'] = global_centroids[:,2]
    return zplanes

def preprocess_tile(tile: da.Array, params:PixelBasedDecodeParams):
    """
    Preprocess a single tile.
    """
    # high pass filter
    if params.high_pass_sigma > 0:
        tile = imagestack_highpass_filter(tile, high_pass_sigma=params.high_pass_sigma, in_place=True)
    # deconvolve
    if params.do_deconv:
        if params.decon_sigma > 0:
            tile = imagestack_deconvolve(tile,  decon_sigma=params.decon_sigma, decon_filter_size=params.decon_filter_size, decon_iterations=params.decon_iterations, in_place=True)
        # blur again
        if params.low_pass_sigma > 0:
            tile = imagestack_lowpass_filter(tile, low_pass_sigma=params.low_pass_sigma, in_place=True)
    return tile

def decode_and_extract_zplane(tile_idx:int,
                z_idx:int,
                data: da.Array,
                params:PixelBasedDecodeParams,
                decoding_matrix:np.ndarray,
                scale_factors:Optional[np.ndarray]=None,
                backgrounds:Optional[np.ndarray]=None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:

    decoded_image, pixel_magnitudes, normalized_pixel_traces, dist = decode_pixels_singleplane_fast(data, decoding_matrix,
                                                                                                    scale_factors=scale_factors, backgrounds=backgrounds)
    crop_width = params.crop_fraction*decoded_image.shape[1]

    # compute everything at once
    decoded_image, pixel_magnitudes, normalized_pixel_traces, dist = da.compute([decoded_image, pixel_magnitudes, normalized_pixel_traces, dist])[0]
    for i in range(pixel_magnitudes.shape[0]):
        decoded_image[i][dist[i] > params.distance_threshold] = -1
        decoded_image[i][pixel_magnitudes[i] < params.magnitude_threshold] = -1

    mols = extract_molecules_from_decoded_image(decoded_image, pixel_magnitudes, normalized_pixel_traces, dist, tile_idx, crop_width, z_idx, params.min_area_size)
    del normalized_pixel_traces
    return mols, decoded_image, pixel_magnitudes, dist

def decode_tile(tile_idx:int,
                tile: TileLoader,
                params:PixelBasedDecodeParams,
                global_align:SimpleGlobalAlignment,
                decoding_matrix:np.ndarray,
                scale_factors:np.ndarray,
                backgrounds:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Registers, preprocesses, and decodes a single tile.
    """
    data, shifts = tile.get_registered_data()
    #data = tile.get_flat_data()
    #fiducial_tile = tile.get_fiducials()
    #ref_img = fiducial_tile[fiducial_tile.coords['image_idx'] == fiducial_tile.coords['image_idx'].values.min()]
    #print(data.shape)
    #print(ref_img.shape)
    #print(fiducial_tile.shape)
    #data, shifts, _ = register_tile_to_reference(tile, fiducial_tile, ref_img, return_reg_fiducials=False)

    # unroll into the correct shape
    #data = unroll_tile(data, coords=['readout_name', 'z'], swap_dims=True, rechunk=False)

    # preoprocess the tile -- using just dask array, chunked as (1,1,n_x,n_y)
    data = data.data#da.from_array(data.data.compute()) # get just the dask array with numpy backing
    n_z,n_bit,n_x,n_y = data.shape
    data = data.rechunk((1,1,n_x,n_y)) # already chunked this way

    data = preprocess_tile(data, params)
    results = da.compute([delayed(decode_and_extract_zplane)(tile_idx, i, data[i], params, decoding_matrix, scale_factors, backgrounds) for i in range(n_z)])[0]

    # try to clean up data
    del data

    mols = [r[0] for r in results]
    decoded_image = np.stack([r[1] for r in results])
    pixel_magnitudes = np.stack([r[2] for r in results])
    dist = np.stack([r[3] for r in results])

    z_pos = np.arange(n_z)
    for i in range(n_z):
        mols[i]['z'] = z_pos[i]#self.z_pos[i] # type: ignore
    mols = pd.concat(mols, ignore_index=True)  # type:ignore
    mols['fov'] = tile_idx
    if global_align is not None:
        centroids = np.array(mols[["z","x","y"]].values)
        global_centroids = global_align.fov_coordinate_array_to_global(tile_idx, centroids)
        mols['global_z'] = global_centroids[:,0]
        mols['global_x'] = global_centroids[:,1]
        mols['global_y'] = global_centroids[:,2]

    # decode the tile
    #decoded_image, pixel_magnitudes, normalized_pixel_traces, dist = imagestack_decode_pixels_dask(data, decoding_matrix,
    #                                                                                          scale_factors=scale_factors, backgrounds=backgrounds,
    #                                                                                          distance_threshold=params.distance_threshold,
    #                                                                                           magnitude_threshold=params.magnitude_threshold)
    # original version returned here

    # new version pushes the molecule extraction into this function
    #mols = extract_molecules_from_tile(tile_idx, decoded_image, dist, normalized_pixel_traces, pixel_magnitudes, params, global_align)
    return shifts, mols, decoded_image, pixel_magnitudes, dist
