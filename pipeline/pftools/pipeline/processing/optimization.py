from typing import Tuple, List, Optional
import dask.array as da
import dask.distributed as dd
import numpy as np
import pandas as pd
import xarray as xr
import logging
import matplotlib.pyplot as plt

from tqdm import tqdm
from pftools.pipeline.core.codebook import Codebook
from pftools.pipeline.util.preprocessing import imagestack_compute_pixel_histogram, imagestack_highpass_filter, imagestack_deconvolve, imagestack_lowpass_filter
from dask.delayed import delayed
from pftools.pipeline.core.tile import TileLoader, get_tile_num_zplanes, unroll_tile
from pftools.pipeline.util.utils import scatter_gather
from pftools.pipeline.processing.decoding import decode_pixels_singleplane_fast, imagestack_decode_pixels
from pftools.pipeline.util.decodeutil import extract_refactors
from pftools.pipeline.util.moleculeutil import extract_molecules_from_decoded_image
from pftools.pipeline.processing.registration import register_tile_by_image_index, imagestack_fiducial_correlation_warp, apply_shifts_by_image_index
from pftools.pipeline.processing.decoding import calculate_normalized_barcodes
from pftools.pipeline.util.barcode_classifier import predict_molecule_quality, calculate_threshold_for_misidentification_rate_ml, calculate_misidentification_rate_for_threshold
from pftools.pipeline.util.decodeutil import extract_refactors_from_molecules

def preprocess_tiles_for_optimization(tiles:List[xr.DataArray],
                                      decon_filter_size:Optional[int]=9,
                                      decon_sigma:int=2,
                                      decon_iterations:int=10, 
                                      high_pass_sigma:int=3,
                                      low_pass_sigma:int=1,
                                      do_deconv:bool=True) -> List[da.Array]:#Tuple[List[da.Array], List[da.Array]]:
    """
    Preprocess the tiles for decoding by deconvoling and computing histogram for initial scale factors.
    Assumes that the tiles are already registered and each tile is of shape Nz x Nbit x Nx x Ny, with the bits in the same order
    as the decoding matrix.

    Returns just the dask arrays for subsequent processing.
    """
    _,_,n_x, n_y = tiles[0].shape
    # rechunk the tiles so that each image is processed independently
    if tiles[0].chunks[0] != 1 and tiles[0].chunks[1] != 1: # type: ignore
        tiles = [t.data.rechunk((1,1,n_x, n_y)) for t in tiles]

    # deconvolve the tiles
    # using dask distributed
    processed_tiles = [imagestack_highpass_filter(t, high_pass_sigma=high_pass_sigma).astype(np.float32) for t in tiles]
    if do_deconv:
        processed_tiles = [imagestack_deconvolve(t, decon_sigma=decon_sigma, decon_filter_size=decon_filter_size, decon_iterations=decon_iterations).astype(np.uint16) for t in processed_tiles]
        processed_tiles = [imagestack_lowpass_filter(t, low_pass_sigma=low_pass_sigma).astype(np.float32) for t in processed_tiles] 
    return processed_tiles

def _get_data_helper(t:TileLoader) -> xr.DataArray:
    return t.get_registered_data()[0]


def get_tiles_for_optimization(client: dd.Client, tiles:List[xr.DataArray], n_tiles=50, random_seed=42) -> Tuple[List[int], List[xr.DataArray]]:
    """
    Select a single tile per fov for optimization.
    """
    np.random.seed(random_seed)
    if n_tiles > len(tiles):
        n_tiles = len(tiles)

    # select a random set of tiles
    tile_idx = np.random.choice(np.arange(len(tiles)), n_tiles, replace=False)#np.random.randint(0, len(tiles), n_tiles)
    tiles_for_optimization = [tiles[i] for i in tile_idx]

    # select a single z plane per fov
    z_pos = tiles_for_optimization[0].get_zpos()
    n_z = tiles_for_optimization[0].get_nz()
    z_pos = [z_pos[np.random.randint(0, n_z)] for t in tiles_for_optimization]

    # get the actual tiles
    tiles_for_optimization = client.gather([client.submit(_get_data_helper, t) for t in tiles_for_optimization])

    # return just the tile, ignoring the shifts
    tiles_for_optimization = [t[t.coords['z']==z] for t, z in zip(tiles_for_optimization, z_pos)]
    return tile_idx, tiles_for_optimization

#def register_fiducials(client: dd.Client, fiducial_tiles:List[xr.DataArray], reference_chan:Optional[str]=None) -> List[xr.DataArray]:
#    """
#    Register the fiducials for the tiles.
#    """
#    # get tiles for optimization
#    # compute registration in parallel
#    if reference_chan is None:
#        reference_idx = 0
#    else:
#        # find the index of the reference channel
#        reference_idx = np.where(fiducial_tiles[0].coords['name'].values==reference_chan)[0][0]

#    results = [client.submit(imagestack_fiducial_correlation_warp, i.data, return_reg=False, reference_idx=reference_idx) for i in fiducial_tiles]
#    shifts = [r[1] for r in client.gather(results)]
#    reg_fids = [apply_shifts_by_image_index(t, shifts[i], image_idx=fiducial_tiles[0].coords['image_idx'].values) for i,t in enumerate(fiducial_tiles)]
#    return reg_fids

def select_tiles_for_optimization(client: dd.Client,
                                               tiles:List[xr.DataArray], 
                                               n_tiles:int=50, random_seed:int=1337) -> List[xr.DataArray]:
    """
    NOTE: This just registers to 0th round -- not to the reference channel. 
    """
    # get tiles for optimization -- registering them before taking a z slice
    tile_idx_opt, tiles_opt = get_tiles_for_optimization(client, tiles, n_tiles=n_tiles,random_seed=random_seed)

    # get fiducials for those tiles
    #fid_tiles_opt = [client.scatter(fiducial_tiles[i].data) for i in tile_idx_opt]
    #results = [client.submit(imagestack_fiducial_correlation_warp, i, return_reg=False) for i in fid_tiles_opt]
    #shifts = [r[1] for r in client.gather(results)]
    #shifts = [r[1] for r in client.gather(results)]
    # XXX dont do this for now -- assume have the same fiducial list
    # subset the fiducials to only the image_idx in the tiles
    #image_idx_used = tiles_opt[0].coords['image_idx'].values
    #fid_tiles_opt = [f[f.coords['image_idx'].isin(image_idx_used)] for f in fid_tiles_opt]
    # compute registration in parallel
    #futures = [client.submit(register_tile_by_image_index, f, t, ) for f,t in zip(fid_tiles_opt, tiles_opt)]
    #reg_tiles_opt = [reg for reg,_ in client.gather(futures)]

    # this was the function used before
    #reg_tiles_opt = [apply_shifts_by_image_index(t, shifts[i], image_idx=fiducial_tiles[0].coords['image_idx'].values) for i,t in enumerate(tiles_opt)]
    # reshape for optimization as Nz x Nround x Nx x Ny
    #reg_tiles_opt = [unroll_tile(t, coords=['readout_name', 'z'], swap_dims=True, rechunk=True) for t in reg_tiles_opt]
    return tiles_opt

def calculate_quantile_across_rounds(img:np.ndarray, quantile:float) -> np.ndarray:
    img = np.squeeze(img)
    n_round, _, _ = img.shape
    return np.array([np.quantile(img[i], quantile) for i in range(n_round)])

def calculate_median_across_rounds(img:np.ndarray) -> np.ndarray:
    meds = []

    img = np.squeeze(img)
    n_round = img.shape[0]
    for i in range(n_round):
        curr_im = img[i]
        meds.append(np.median(curr_im[curr_im>0]))
    return np.array(meds)

def calculate_initial_scale_factors(imgs:np.ndarray, fg_quantile=0.95, bg_quantile=0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the initial scale factors from the pixel histograms.
    """
    #initial_scale_factors = np.zeros(n_bits)
    #for i in range(n_bits):
    #    cumulative_histogram = np.cumsum(histograms[i]).astype(np.float32)
    #    cumulative_histogram = cumulative_histogram/cumulative_histogram[-1]
    #    initial_scale_factors[i] = np.argmin(np.abs(cumulative_histogram-0.9))+2
    #return initial_scale_factors
    fg_scales = np.mean(np.array([calculate_quantile_across_rounds(i, fg_quantile) for i in imgs]),axis=0)
    #fg_scales = np.mean(np.array([calculate_median_across_rounds(i) for i in imgs]),axis=0)
    bg_scales = np.mean(np.array([calculate_quantile_across_rounds(i, bg_quantile) for i in imgs]), axis=0)
    return fg_scales, bg_scales

def update_scale_factors(scale_factors:np.ndarray, backgrounds:np.ndarray,
                         scale_refactors:List[np.ndarray], background_refactors:List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        n_tiles = len(scale_refactors)
        prev_scale_factors = np.repeat(np.expand_dims(scale_factors,0), n_tiles, axis=0)
        prev_backgrounds = np.repeat(np.expand_dims(backgrounds,0), n_tiles, axis=0)

        # find the median scale_factor and background across all FOV, scaled by the previous scale factors
        curr_scale_factors = np.nanmedian(np.multiply(scale_refactors, prev_scale_factors), axis=0)
        # scale the backgrounds by the overall scale factor
        curr_backgrounds = np.nanmedian(np.add(prev_backgrounds, np.multiply(background_refactors, prev_scale_factors)), axis=0)
        return curr_scale_factors, curr_backgrounds

def init_scale_factors(n_iter:int, n_bit:int, n_tiles:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    previous_scale_factors = np.zeros((n_iter, n_bit))
    previous_backgrounds = np.zeros_like(previous_scale_factors)
    scale_refactors = np.zeros((n_iter, n_tiles, n_bit))
    background_refactors = np.zeros_like(scale_refactors)
    return previous_scale_factors, previous_backgrounds, scale_refactors, background_refactors

def optimize_chromatic_aberration_across_tiles(client:dd.Client, 
                                               tiles:List[xr.DataArray],
                                               codebook:Codebook,
                                               magnitude_threshold:float=1,
                                               distance_threshold:float=0.65,
                                               area_threshold:int=4,
                                               decon_sigma:float=2,
                                               decon_iterations:int=10,
                                               decon_filter_size:Optional[int]=9,
                                               high_pass_sigma:int=3,
                                               low_pass_sigma:float=2,
                                               n_iter:int=10,
                                               logger:Optional[logging.Logger]=None,
                                               do_deconv:bool=True) -> List[np.ndarray]:
    """
    Optimize chromatic aberration by first decoding, and then finding transformation that aligns bits across different colors. 
    """
    decoding_matrix = calculate_normalized_barcodes(codebook)
    n_bits = decoding_matrix.shape[1]
    n_tiles = len(tiles)
    if logger is not None:
        logger.info("Setting up preprocessing before chromatic aberration correction...")
    preprocessed_tiles = preprocess_tiles_for_optimization(tiles,
                                                        low_pass_sigma=low_pass_sigma, 
                                                        high_pass_sigma=high_pass_sigma,
                                                        decon_filter_size=decon_filter_size, 
                                                        decon_sigma=decon_sigma, 
                                                        decon_iterations=decon_iterations,
                                                        do_deconv=do_deconv)
    if logger is not None:
        logger.info("Preprocessing tiles before optimization...")
    preprocessed_tiles = da.compute([t for t in preprocessed_tiles])[0]
    if logger is not None:
        logger.info(f"Found {len(preprocessed_tiles)} tiles")
    preprocessed_tiles = [np.squeeze(t).astype(np.float32) for t in preprocessed_tiles]
    del tiles
    #histograms = [h.compute() for h in histograms]

    # load these onto the cluster
    if logger is not None:
        logger.info("Loading tiles onto cluster...")

    #preprocessed_tiles = client.scatter(preprocessed_tiles)
    # preprocess the data
    if logger is not None:
        logger.info("Optimizing scale factors...")
        iterator = range(n_iter)
    else:
        iterator = tqdm(range(n_iter))

    # TODO Implement the rest of this

def optimize_scalefactors_across_tiles(client:dd.Client,
                                     tiles:List[xr.DataArray],
                                     codebook:Codebook,
                                     magnitude_threshold:float=1,
                                     distance_threshold:float=0.65,
                                     area_threshold:int=4,
                                     decon_sigma:float=2,
                                     decon_iterations:int=10,
                                     decon_filter_size:Optional[int]=9,
                                     high_pass_sigma:int=3,
                                     low_pass_sigma:float=2,
                                     extract_backgrounds:bool=True,
                                     n_iter:int=10,
                                     logger:Optional[logging.Logger]=None,
                                     do_deconv:bool=True,
                                     background_quantile:float=0.05,
                                     foreground_quantile:float=0.95,
                                     use_ml_filt:bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize the scale factors across a chosen set of FOVs.
    """
    decoding_matrix = calculate_normalized_barcodes(codebook)
    # preprocess the tiles -- get deconvolved, and histogram for each tile
    n_bit = decoding_matrix.shape[1]
    n_tiles = len(tiles)

    if logger is not None:
        logger.info("Setting up preprocessing before optimization...")
    preprocessed_tiles = preprocess_tiles_for_optimization(tiles,
                                                            low_pass_sigma=low_pass_sigma, 
                                                            high_pass_sigma=high_pass_sigma,
                                                            decon_filter_size=decon_filter_size, 
                                                            decon_sigma=decon_sigma, 
                                                            decon_iterations=decon_iterations,
                                                            do_deconv=do_deconv)

    # do low pass filtering here
    # just using dask
    #preprocessed_tiles = [imagestack_lowpass_filter(t, low_pass_sigma=low_pass_sigma) for t in preprocessed_tiles]
    # using dask distributed
    #preprocessed_tiles = scatter_gather(client, preprocessed_tiles, imagestack_lowpass_filter, low_pass_sigma=low_pass_sigma)

    # compute these before running deconv -- now just a list of numpy arrays
    if logger is not None:
        logger.info("Preprocessing tiles before optimization...")
    preprocessed_tiles = da.compute([t for t in preprocessed_tiles])[0]
    if logger is not None:
        logger.info(f"Found {len(preprocessed_tiles)} tiles")
    preprocessed_tiles = [np.squeeze(t).astype(np.float32) for t in preprocessed_tiles]
    del tiles
    #histograms = [h.compute() for h in histograms]

    # load these onto the cluster
    if logger is not None:
        logger.info("Loading tiles onto cluster...")

    #preprocessed_tiles = client.scatter(preprocessed_tiles)
    # preprocess the data
    if logger is not None:
        logger.info("Optimizing scale factors...")
        iterator = range(n_iter)
    else:
        iterator = tqdm(range(n_iter))

    previous_scale_factors, previous_backgrounds, scale_refactors, background_refactors = init_scale_factors(n_iter, n_bit, n_tiles)
    barcode_counts = []

    for i in iterator:
        if logger is not None:
            logger.info("Optimization iteration %d..." % i)

        # on each iteration, just needs previous scale factors
        if i == 0:
            curr_scale_factors, curr_backgrounds = calculate_initial_scale_factors(preprocessed_tiles, 
                                                                                   fg_quantile=foreground_quantile,
                                                                                   bg_quantile=background_quantile)
            previous_scale_factors[i] = curr_scale_factors
            previous_backgrounds[i] = curr_backgrounds

        else:
            refactors = scale_refactors[i-1,:,:]
            refactors[refactors==0] = 1
            bg_refactors = background_refactors[i-1,:,:]
            prev_scale_factors = np.repeat(np.expand_dims(previous_scale_factors[i-1,:],0), n_tiles, axis=0)
            prev_backgrounds = np.repeat(np.expand_dims(previous_backgrounds[i-1,:],0), n_tiles, axis=0)

            # find the median scale_factor and background across all FOV, scaled by the previous scale factors
            curr_scale_factors = np.nanmedian(refactors * prev_scale_factors, axis=0)#np.nanmedian(np.multiply(refactors, prev_scale_factors), axis=0)
            # scale the backgrounds by the overall scale factor
            curr_backgrounds = np.nanmedian(prev_backgrounds + (bg_refactors * prev_scale_factors),axis=0)#np.nanmedian(np.add(prev_backgrounds, np.multiply(bg_refactors, prev_scale_factors)), axis=0)
            previous_scale_factors[i] = curr_scale_factors
            previous_backgrounds[i] = curr_backgrounds
        if use_ml_filt:
            if logger is not None:
                logger.info("Decoding and extracting barcodes...")
            results = scatter_gather(client, preprocessed_tiles, optimize_decode_and_extract_barcodes, decoding_matrix, curr_scale_factors, curr_backgrounds, distance_threshold, magnitude_threshold, area_threshold)
            #results = optimize_extract_raw_pixel_traces(preprocessed_tiles, results)
            results = pd.concat(results,axis=0) # combine all molecules
            results = results[results.barcode_id != -1] # remove molecules that were not decoded
            results = results[results.barcode_id != 65535] # remove molecules that were not decoded
            if logger is not None:
                logger.info("Classifying molecules and extracting scale factors...")
            bs, refactors, backgrounds = optimize_extract_scalefactors_from_molecules(results, codebook, misidentification_rate=0.05, extract_backgrounds=extract_backgrounds)
        else:
            results = scatter_gather(client, preprocessed_tiles, imagestack_optimize_scalefactors, decoding_matrix, curr_scale_factors, curr_backgrounds, distance_threshold, magnitude_threshold, extract_backgrounds, area_threshold)
            #results = da.compute([delayed(imagestack_optimize_scalefactors)(t, decoding_matrix, curr_scale_factors, curr_backgrounds) for t in preprocessed_tiles])[0]
            bs, refactors, backgrounds = zip(*results)
        barcode_counts.append(bs) 
        scale_refactors[i,:,:] = np.array(refactors)
        background_refactors[i,:,:] = np.array(backgrounds)
        
    return np.array(barcode_counts), np.array(previous_scale_factors), np.array(previous_backgrounds), np.array(scale_refactors), np.array(background_refactors)

def optimize_extract_scalefactors_from_molecules(mols:pd.DataFrame, codebook:Codebook, misidentification_rate:float=0.05, extract_backgrounds:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mols = mols[(mols.min_distance < 0.5)&(mols.area>4)]
    #mols = predict_molecule_quality(mols, codebook)
    #opt_thresh = calculate_threshold_for_misidentification_rate_ml(mols, codebook, misidentification_rate)
    #thresh = 0.2 # use only the really good molecules #calculate_threshold_for_misidentification_rate(mols, codebook, misidentification_rate)

    #plt.figure()
    #plt.hist(mols[(mols.quality < thresh)&(mols.is_coding)].min_distance, bins=100, density=True, alpha=0.5)
    #plt.hist(mols[(~mols.is_coding)].min_distance, bins=100, density=True, alpha=0.5)
    #plt.xlim([0, 0.8])
    #print("Useful molecules", np.sum((mols.quality < thresh)&(mols.is_coding)), "out of", np.sum(mols.is_coding))
    #print("Mean coding distance", np.mean(mols[(mols.quality < thresh)&(mols.is_coding)].min_distance))
    #print("Mean noncoding distance", np.mean(mols[(~mols.is_coding)].min_distance))
    #print("Misidentification rate", calculate_misidentification_rate_for_threshold(mols, len(codebook.get_coding_indexes()), len(codebook.get_blank_indexes()), thresh))
    #print("Opt thresh", opt_thresh)
    #mols = mols[(mols['quality'] < thresh) & (mols.is_coding)]

    decoding_matrix = calculate_normalized_barcodes(codebook)
    refactors, background_refactors, barcodes_seen = extract_refactors_from_molecules(mols, decoding_matrix, extract_backgrounds=extract_backgrounds)
    return barcodes_seen, refactors, background_refactors

def optimize_extract_raw_pixel_traces(images:List[np.ndarray], mols:List[pd.DataFrame], n_pix:int=2) -> pd.DataFrame:
    # extract raw pixel traces from a window around each molecule
    # XXX not currently used
    n_bits = images[0].shape[0]
    colnames = ['raw_intensity_%d' % i for i in range(n_bits)]
    for i in range(len(images)):
        curr_img = images[i]
        curr_mol = mols[i]
        intensities = []
        for idx, row in curr_mol.iterrows():
            x,y = row['x'], row['y']
            x_min = int(max(0, x-n_pix))
            x_max = int(min(curr_img.shape[1], x+n_pix+1))
            y_min = int(max(0, y-n_pix))
            y_max = int(min(curr_img.shape[2], y+n_pix+1))
            curr_intensities = curr_img[:,x_min:x_max, y_min:y_max].reshape(n_bits, -1).mean(axis=1)
            intensities.append(curr_intensities) 
        intensities = np.vstack(intensities)
        mols[i] = pd.concat([curr_mol, pd.DataFrame(intensities, index=curr_mol.index, columns=colnames)], axis=1)
    return mols

def optimize_decode_and_extract_barcodes(images:np.ndarray, 
                                        decoding_matrix:np.ndarray, 
                                        scale_factors:np.ndarray, 
                                        backgrounds:np.ndarray,
                                        distance_threshold:float=1, 
                                        magnitude_threshold:float=0.25,
                                        area_threshold:float=5) -> pd.DataFrame:
    di, pm, npt, d = decode_pixels_singleplane_fast(np.squeeze(images).astype(np.float32), decoding_matrix, scale_factors, backgrounds) 
    di[d > distance_threshold] = -1
    di[pm < magnitude_threshold] = -1
    crop_width = 0.05 * di.shape[1]

    mols = extract_molecules_from_decoded_image(di, pm, npt, d, 0, crop_width, 0, area_threshold)
    del di, pm, npt, d

    return mols
 

def imagestack_optimize_scalefactors(images:np.ndarray,
                                     decoding_matrix:np.ndarray,
                                     scale_factors:np.ndarray, backgrounds:np.ndarray,
                                    distance_threshold:float=0.5176, magnitude_threshold:float=1,
                                    extract_backgrounds:bool=True,
                                    area_threshold:float=4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Iteratively compute the scale factors used for decoding.
    Inputs:
        images: ndarray of shape (Nz, Nround, Nx, Ny)
        decoding_matrix: np.ndarray of shape (Nbit, Ncode)
        n_iter: number of iterations to run
        chromatic_aberration_correction: whether to correct for chromatic aberration
        color_channels: list of color channels to use for chromatic aberration correction
        distance_threshold: distance threshold for decoding
        magnitude_threshold: magnitude threshold for decoding
        area_threshold: area threshold for decoding
    Outputs:
        barcodes: pd.DataFrame with barcodes
        scale_factors: np.ndarray of shape (Nround,)
        backgrounds: np.ndarray of shape (Nround,)
    """
    di, pm, npt, d = decode_pixels_singleplane_fast(np.squeeze(images).astype(np.float32), decoding_matrix, scale_factors, backgrounds) 
    di[d > distance_threshold] = -1
    di[pm < magnitude_threshold] = -1

    #di, pm, npt, _ = imagestack_decode_pixels(images, decoding_matrix, scale_factors, backgrounds, distance_threshold, magnitude_threshold)
    #di = np.squeeze(di)
    #pm = np.squeeze(pm)
    #npt = np.squeeze(npt)
    # get rid of extra dimension for these and compute result
    refactors, backgrounds, barcodes_seen = extract_refactors(decoding_matrix, di,  pm, npt, extractBackgrounds=extract_backgrounds, refactor_area_threshold=area_threshold)
    return barcodes_seen, refactors, backgrounds
