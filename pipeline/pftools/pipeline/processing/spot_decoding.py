import xarray as xr
from typing import Optional, List, Tuple, Dict
import dask.distributed as dd
import numpy as np
import dask.array as da
from scipy.stats import zscore
import dask.distributed as dd
import logging
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass

from pftools.pipeline.core.codebook import Codebook
from pftools.pipeline.processing.globalalign import SimpleGlobalAlignment
from pftools.pipeline.util.spotutils import find_spots_across_rounds
from pftools.pipeline.core.algorithm import ParallelTileAnalysisTask
from pftools.pipeline.util.spotutils import extract_pixel_traces_for_spots, filter_neighbors_3d
from pftools.pipeline.processing.decoding import calculate_normalized_barcodes
from pftools.pipeline.util.preprocessing import imagestack_highpass_filter
from pftools.pipeline.processing.decoding import calculate_normalized_barcodes
from pftools.pipeline.util.moleculeutil import molecule_count_histograms_spots, blank_fraction_histogram, calculate_threshold_for_misidentification_rate, crop_molecules, assign_gene_symbols_to_mols
from pftools.pipeline.util.decodeutil import calculate_normalized_barcodes
from pftools.pipeline.core.tile import TileLoader

@dataclass
class SpotDecoderParams:
    distance_thresh: float=0.65
    magnitude_threshold: float=1
    high_pass_sigma: float=5
    misidentification_rate:float=0.05
    crop_fraction:float=0.00 # originally 0.05

class SpotDecoder(ParallelTileAnalysisTask):
    def __init__(self, 
                 client:dd.Client, 
                 tiles: List[TileLoader],
                 output_path:str, 
                 codebook:Codebook,
                 params:Optional[SpotDecoderParams]=None, 
                 global_align:Optional[SimpleGlobalAlignment]=None):
        super().__init__(client, tiles, output_path)
        self.codebook = codebook
        self.global_align = global_align
        self.tiles = tiles
        self.pos = None
        self.spot_traces = None
        if params is None:
            params = SpotDecoderParams()
        self.params = params
        self.barcodes = {}
        self.spots = {}

    def _tile_processing_done_callback(self, tile_idx: int, future: dd.Future) -> None:
        #self.logger.info(f"Finished processing tile {tile_idx}")
        if future is not None:
            spots, bc = future.result() 
            del future # try to make sure everything is deleted from distributed RAM
            # save out the spots and barcodes
            self._save_spots_and_barcodes(tile_idx, spots, bc)
        else:
            self.logger.info("Loading spots and barcodes from tile %d" % tile_idx)
            data = np.load(os.path.join(self.output_path, 'spots', f'spots_{tile_idx}.npy.npz'))
            spots = data['spots']
            bc = data['bc']
        self.barcodes[tile_idx] = bc
        self.spots[tile_idx] = spots

    def _save_spots_and_barcodes(self, tile_idx:int, spots:np.ndarray, bc:np.ndarray) -> None:
        spots_path = os.path.join(self.output_path, 'spots')
        if not os.path.exists(spots_path):
            os.makedirs(spots_path)
        np.savez(os.path.join(spots_path, f'spots_{tile_idx}.npy'), spots=spots, bc=bc)

    def _postrun_processing(self) -> None:
        # collect all the spots
        self.logger.info("Collecting all spots")
        spots, bc, fov = self._collect_spots()

        # optimize the scale factors
        self.logger.info("Optimizing scale factors")
        bc_mat = calculate_normalized_barcodes(self.codebook)
        bc_mat_coding = bc_mat[self.codebook.get_coding_indexes(),:]
        np.savez(os.path.join(self.output_path, 'raw_bc.npz'), spots=spots, bc=bc, fov=fov)

        sf, bg, ps, pb = optimize_scale_factors(bc_mat_coding, bc, n_iter=10)

        np.save(os.path.join(self.output_path, 'scale_factors.npy'), sf)
        np.save(os.path.join(self.output_path, 'backgrounds.npy'), bg)
        np.save(os.path.join(self.output_path, 'previous_scale.npy'), np.array(ps))
        np.save(os.path.join(self.output_path, 'previous_background.npy'), np.array(pb))
        # decode spots
        self.logger.info("Decoding spots")
        decoded, distances, pixel_magnitudes, normalized_pixel_traces = decode_spots(bc_mat, bc, scale_factors=sf, backgrounds=bg, 
                                                                                     distance_thresh=self.params.distance_thresh)
        np.save(os.path.join(self.output_path, 'decoded.npy'), decoded)
        np.save(os.path.join(self.output_path, 'distances.npy'), distances)
        np.save(os.path.join(self.output_path, 'pixel_magnitudes.npy'), pixel_magnitudes)
        np.save(os.path.join(self.output_path, 'normalized_pixel_traces.npy'), normalized_pixel_traces)

        # filter out spots that aren't good
        self.logger.info("Filtering spots")
        # find misidentification threshold cutoff
        bad_spots = np.argwhere((distances > self.params.distance_thresh) | (pixel_magnitudes < self.params.magnitude_threshold)).ravel()
        decoded[bad_spots] = -1

        # converting to standardized format for export
        mols = self._reformat_molecules(fov, spots, decoded, distances, pixel_magnitudes, normalized_pixel_traces)
        mols = mols[mols.barcode_id != -1]
        mols = assign_gene_symbols_to_mols(mols, self.codebook)
        mols.to_parquet(os.path.join(self.output_path, 'molecules.parquet'), index=False)

        # dynamically filter molecules and crop down to smaler FOV
        mols_filt = self._dynamic_filter_molecules(mols)
        self.logger.info(f"Keeping {mols_filt.shape[0]} ({mols_filt.shape[0]/mols.shape[0]*100:.2f}%) molecules")

        #nx, ny = self.tiles[0].get_flat_data().shape[1:]
        #self.logger.info(f"Cropping down to {self.params.crop_fraction*100}% ({(1-self.params.crop_fraction)*nx} x {(1-self.params.crop_fraction)*ny}) of FOV")
        #mols_filt = crop_molecules(mols_filt, self.params.crop_fraction, nx, ny)
        mols_filt.to_parquet(os.path.join(self.output_path, 'filtered_molecules.parquet'), index=False)

    def _dynamic_filter_molecules(self, mols:pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Filtering molecules")
        self.logger.info(f"Calculating threshold for misidentification rate {self.params.misidentification_rate}")
        n_coding = len(self.codebook.get_coding_indexes())
        n_blank = len(self.codebook.get_blank_indexes())
        coding_counts, blank_counts, intensity_bins, distance_bins = molecule_count_histograms_spots(mols, self.codebook, dist_thresh=self.params.distance_thresh)
        blank_frac = blank_fraction_histogram(coding_counts, blank_counts, n_coding, n_blank)
        threshold = calculate_threshold_for_misidentification_rate(self.params.misidentification_rate, coding_counts, blank_counts, blank_frac, n_coding, n_blank)
        self.logger.info(f"Identified threshold of {threshold}")
        select_data = mols[["mean_intensity", "min_distance"]].values
        select_data[:,0] = np.log10(select_data[:,0])
        barcode_bins = np.array(
            (np.digitize(select_data[:,0], intensity_bins, right=True),
            np.digitize(select_data[:,1], distance_bins, right=True)))
        barcode_bins[0,:] = np.clip(barcode_bins[0,:], 0, blank_frac.shape[0]-1)
        barcode_bins[1,:] = np.clip(barcode_bins[1,:], 0, blank_frac.shape[1]-1)
        raveled_indexes = np.ravel_multi_index(barcode_bins[:, :], blank_frac.shape)
        thresholded_blank_frac = blank_frac < threshold
        return mols[np.take(thresholded_blank_frac, raveled_indexes)]

    def _collect_spots(self):        
        #self.barcodes[tile_idx] = bc
        #self.spots[tile_idx] = spots
        nfov = len(self.tiles)
        all_spots = []
        all_fov = []
        for tile_idx in range(nfov):
            spot_path = os.path.join(self.output_path, 'spots', f'spots_{tile_idx}.npy.npz')
            if os.path.exists(spot_path):
                data = np.load(spot_path)
                spots = data['spots']
                bc = data['bc']
                self.barcodes[tile_idx] = bc
                self.spots[tile_idx] = spots

        for f, s in self.spots.items():
            all_spots.append(s)
            all_fov.append(np.ones((s.shape[0],))*f)
        all_spots = np.vstack(all_spots)
        all_bc = np.vstack([v for v in self.barcodes.values()])
        all_fov = np.hstack(all_fov)
        return all_spots, all_bc, all_fov

    def _reformat_molecules(self, fov:np.ndarray, spots:np.ndarray, decoded:np.ndarray, distances:np.ndarray, magnitudes:np.ndarray, normalized_pixel_traces:np.ndarray) -> pd.DataFrame:
        intensity_columns = ['intensity_%d' % i for i in range(normalized_pixel_traces.shape[1])]
        intensities = pd.DataFrame(normalized_pixel_traces, columns=intensity_columns)
        mols = pd.DataFrame({'fov': fov, 
                             'z': spots[:,0],
                             'x': spots[:,1],
                             'y': spots[:,2],
                             'barcode_id' : decoded,
                             'mean_distance' : distances,
                             'min_distance' : distances,
                             'mean_intensity' : magnitudes,
                             'max_intensity': magnitudes,
                             })
        mols['cell_index'] = -1
        if self.global_align is not None:
            for f in mols.fov.unique():
                curr_fov = mols[mols.fov==f]
                pos = np.array(curr_fov[["z","y","x"]].values)
                global_pos = self.global_align.fov_coordinate_array_to_global(int(f), pos)
                mols.loc[mols.fov==f, ['global_z','global_x','global_y']] = global_pos
            #globalpos = 
        mols = pd.concat([mols, intensities], axis=1)
        return mols

class MultiRoundSpotDecoder(SpotDecoder):
    """
    Identify all spots on every round, and then combine into a single list of spots by identifying the same spot across rounds. 
    """
    def __init__(self, 
                client:dd.Client, 
                tiles: List[TileLoader],
                output_path:str, 
                codebook:Codebook, 
                params: Optional[SpotDecoderParams]=None,
                global_align:Optional[SimpleGlobalAlignment]=None):
        
        super().__init__(client, tiles, output_path, codebook, params, global_align)

    def _process_tile(self, tile_idx: int) -> None:
        spot_path = os.path.join(self.output_path, 'spots', f'spots_{tile_idx}.npy.npz')
        if os.path.exists(spot_path):
            self.logger.info(f"Tile {tile_idx} already processed, skipping")
            return tile_idx, None
        else:
            tile = self.tiles[tile_idx]
            #fid_tile = self.fiducial_tiles[tile_idx]
            tile = self.client.scatter(tile)
            #fid_tile = self.client.scatter(fid_tile)
            # if the reference fiducial is not specified, use the first round
            #if self.reference_fiducials is None:
            #    ref_fid = self.fiducial_tiles[0]
            #    ref_fid = ref_fid[ref_fid.coords['image_idx'] == ref_fid.coords['image_idx'].values.min()][0]
            #else:
            #    ref_fid = self.reference_fiducials[tile_idx]
            #ref_fid = self.client.scatter(ref_fid)
            return tile_idx, self.client.submit(process_spots_multiround, tile_idx=tile_idx, tile=tile, 
                                                high_pass_sigma=self.params.high_pass_sigma, logger=self.logger)
            #return tile_idx, process_spots_multiround(tile_idx=tile_idx, tile=tile, fid_tile=fid_tile, 
        #                                    ref_fid=ref_fid, high_pass_sigma=self.params.high_pass_sigma, logger=self.logger)


class CommonRoundSpotDecoder(SpotDecoder):
    """
    Identify all spots on a reference round, then use for decoding. 
    """
    def __init__(self, client:dd.Client, 
                tiles: List[xr.DataArray],
                fiducial_tiles: List[xr.DataArray],
                output_path:str, 
                codebook:Codebook, 
                global_align:Optional[SimpleGlobalAlignment]=None,
                reference_fiducials: Optional[List[xr.DataArray]]=None):
        super().__init__(client, tiles, fiducial_tiles, output_path, codebook, global_align, reference_fiducials)

    def _process_tile(self, tile_idx: int) -> None:
        tile = self.client.scatter(self.tiles[tile_idx])
        fid_tile = self.client.scatter(self.fiducial_tiles[tile_idx])
        if self.reference_fiducials is None:
            ref_fid = get_reference_fiducial(self.fiducial_tiles[0])
        else:
            ref_fid = self.reference_fiducials[tile_idx]
        ref_fid = self.client.scatter(ref_fid)
        return tile_idx, self.client.submit(process_spots_common_round, tile_idx=tile_idx, tile=tile, fid_tile=fid_tile, 
                                            ref_fid=ref_fid, logger=self.logger)
        #return tile_idx, process_spots_common_round(tile_idx=tile_idx, tile=tile, fid_tile=fid_tile, 
        #                                    ref_fid=ref_fid, logger=self.logger)

def get_reference_fiducial(tiles:List[xr.DataArray], ref_round:int=0) -> xr.DataArray:
    """
    Get the reference fiducial for a set of tiles. 
    """
    ref_fid = tiles[ref_round]
    ref_fid = ref_fid[ref_fid.coords['image_idx'] == ref_fid.coords['image_idx'].values.min()][0]
    return ref_fid

def process_spots_common_round(tile_idx:int, tile: xr.DataArray, fid_tile:xr.DataArray, ref_fid:xr.DataArray, high_pass_sigma:int=3, logger:Optional[logging.Logger]=None) -> Tuple[np.ndarray, np.ndarray]:
    if logger is not None:
        logger.info(f"Pre-processing tile {tile_idx}")
    # register and high pass
    t = preprocess_tile_for_spot_finding(tile, fid_tile, ref_fid, high_pass_sigma=high_pass_sigma)
    t = t.compute().astype(np.uint16)
    del fid_tile
    del tile
    if logger is not None:
        logger.info(f"Finding spots for tile {tile_idx}")
    # only find spots on the first round
    spots, spots_filt = find_spots_for_decoding(t[0])
    # extract traces from subsequent rounds
    bc = extract_spot_traces(t[1:], spots_filt)
    return spots_filt, bc

def process_spots_multiround(tile_idx:int, tile: TileLoader, high_pass_sigma:int=3, logger:Optional[logging.Logger]=None) -> Tuple[np.ndarray, np.ndarray]:
    if logger is not None:
        logger.info(f"Pre-processing tile {tile_idx}")
    t,_ = tile.get_registered_data()

    t = preprocess_tile_for_spot_finding(t, high_pass_sigma=high_pass_sigma)
    # t is a dask array
    t = t.compute().astype(np.uint16)
    if logger is not None:
        logger.info(f"Finding spots for tile {tile_idx}")
    spots, spots_filt = find_spots_for_decoding(t)
    bc = extract_spot_traces(t, spots_filt)
    #cb_mat = calculate_normalized_barcodes(codebook)
    #d, idx = decode_spots(bc_scaled, cb_mat)
    return spots_filt, bc

def preprocess_tile_for_spot_finding(tile: xr.DataArray, high_pass_sigma:int=3) -> da.Array:
    """
    Register and high pass filter a tile for spot finding. 
    """
    #print("registering to reference")
    #reg_tile, shifts = tile.get_registered_data()#register_tile_to_reference(tile, fid_tile, ref_fid, return_reg_fiducials=False)
    #reg_tile = da.from_array(reg_tile.data.compute())
    #reg_tile, shifts, _ = register_tile(tile, fid_tile, return_reg_fiducials=False)
    #print(shifts)
    #t = unroll_tile(reg_tile, coords=['readout_name', 'z'], swap_dims=True, rechunk=False)
    # get dask array
    t = tile.data.astype(np.uint16)
    _, _, n_x, n_y = t.shape
    #t = t.rechunk((1,1,n_x, n_y))
    t = imagestack_highpass_filter(t, high_pass_sigma=high_pass_sigma)
    return t.astype(np.uint16)


def find_spots_for_decoding(arr:np.ndarray, th_seed:float=10, filt_size:int=10, gfilt_size:int=10) -> np.ndarray:
    spots = find_spots_across_rounds(arr, use_fitting=False, th_seed=th_seed, filt_size=filt_size,gfilt_size=gfilt_size)
    # identify a common set of spots across rounds -- don't use height 
    spots_filt = filter_neighbors_3d(spots[:,:-1])
    return spots, spots_filt

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def extract_spot_traces_pool(arr:np.ndarray, spots:np.ndarray, radius:int=1) -> np.ndarray:
    """
    Extract the pixel traces for the spots using max pooling.
    """
    bc = np.zeros((spots.shape[0], arr.shape[1]))
    for i in range(spots.shape[0]):
        min_x = int(max(0, spots[i,1]-radius))
        max_x = int(min(arr.shape[2], spots[i,1]+radius+1))
        min_y = int(max(0, spots[i,2]-radius))
        max_y = int(min(arr.shape[3], spots[i,2]+radius+1))
        bc[i,:] = arr[int(spots[i,0]), :, min_x:max_x, min_y:max_y].mean((1,2))
    return bc

def extract_spot_traces(arr:np.ndarray, spots:np.ndarray, radius:int=2, subtract_bg:bool=False) -> np.ndarray:
    """
    Extract the pixel traces for the spots.
    """
    bc, bg, bg_std = extract_pixel_traces_for_spots(np.swapaxes(arr, 0, 1), spots, subtract_bg=subtract_bg, win_size=radius)
    #for i in range(bc.shape[0]):
        #for j in range(bc.shape[1]):
            #bc[i,j] = (bc[i,j] - bg[i,j])
    #bc -= bg
    #bc_scaled = bc.copy()
    #background = np.quantile(bc, 0.05,axis=0)
    #scale_factor = np.quantile(bc, 0.95, axis=0)
    # normalize readouts
    #for i in range(bc.shape[1]):
    #    bc_scaled[:,i] = zscore(bc_scaled[:,i])#(bc_scaled[:,i] - background[i])/scale_factor[i]

    #bc_mag = np.linalg.norm(bc_scaled,axis=1)
    #for i in range(bc_scaled.shape[0]):
        #bc_scaled[i,:] /= bc_mag[i]
    #    bc_scaled[i,:] = normalize(bc_scaled[i,:])
    return bc#, bc_scaled

def normalize(x):
    norm = np.linalg.norm(x)
    if norm > 0:
        return x/norm
    else:
        return x

#def calculate_normalized_barcodes(codebook):
#    magnitudes = np.sqrt(np.sum(codebook * codebook, axis=1))
#    return np.array([normalize(x) for x, m in zip(codebook, magnitudes)])

def optimize_scale_factors(codebook_matrix: np.ndarray, pixel_traces: np.ndarray,
                           magnitude_thresh: float=0.25, distance_thresh: float=0.65, verbose:bool=True, n_iter:int=10):
    #refactors = get_initial_scale_factors(codebook_matrix, pixel_traces)
    decoded, distances, pixel_magnitudes, normalized_pixel_traces = decode_spots(codebook_matrix, pixel_traces, 
                                                                                 magnitude_thresh=magnitude_thresh, distance_thresh=distance_thresh)
    scale_factors, barcodes_seen = extract_refactors(codebook_matrix, decoded, pixel_magnitudes, normalized_pixel_traces)
    background_scale_factors = extract_backgrounds(codebook_matrix, decoded, pixel_magnitudes, normalized_pixel_traces)
    previous_scale = [scale_factors]
    previous_background = [background_scale_factors]
    #return decoded, distances, pixel_magnitudes, normalized_pixel_traces, previous_scale, previous_background
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(range(n_iter))
    else:
        iterator = range(n_iter)
    for i in iterator:
        decoded, distances, pixel_magnitudes, normalized_pixel_traces = decode_spots(codebook_matrix, pixel_traces, 
                                                                                     scale_factors=scale_factors, backgrounds=background_scale_factors,
                                                                                     magnitude_thresh=magnitude_thresh, distance_thresh=distance_thresh)
        scale_factors, barcodes_seen = extract_refactors(codebook_matrix, decoded, pixel_magnitudes, normalized_pixel_traces)
        background_scale_factors = extract_backgrounds(codebook_matrix, decoded, pixel_magnitudes, normalized_pixel_traces)
        scale_factors[scale_factors == 0] = 1
        scale_factors = np.nanmedian(np.multiply(scale_factors, np.array(previous_scale)), axis=0)
        background_scale_factors = np.nanmedian(np.add(np.array(previous_background), np.multiply(background_scale_factors, np.array(previous_scale))), axis=0)
        previous_scale.append(scale_factors)
        previous_background.append(background_scale_factors)
    return scale_factors, background_scale_factors, previous_scale, previous_background

def decode_spots(codebook_matrix: np.ndarray, pixel_traces: np.ndarray, scale_factors: Optional[np.ndarray]=None, 
                 backgrounds: Optional[np.ndarray]=None, distance_thresh:float=0.57, magnitude_thresh:float=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_bc, n_bits = pixel_traces.shape
    
    if scale_factors is None:
        scale_factors = np.quantile(pixel_traces, 0.95, axis=0)#np.ones((n_bits,))
    if backgrounds is None:
        backgrounds = np.quantile(pixel_traces, 0.05, axis=0) #np.zeros((n_bits,))
     
    scaled_pixel_traces = np.zeros_like(pixel_traces)
    for i in range(n_bc):
        for j in range(n_bits):
            scaled_pixel_traces[i,j] = (pixel_traces[i,j]-backgrounds[j])/scale_factors[j]
            
    pixel_magnitudes = np.array([np.linalg.norm(x) for x in scaled_pixel_traces])
    pixel_magnitudes[pixel_magnitudes == 0] = 1
    
    normalized_pixel_traces = scaled_pixel_traces/pixel_magnitudes[:, None]
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=2)
    neighbors.fit(codebook_matrix)
    distances, decoded = neighbors.kneighbors(normalized_pixel_traces, return_distance=True)
    # do this later
    decoded[distances > distance_thresh] = -1
    decoded[pixel_magnitudes < magnitude_thresh] = -1
    return decoded.ravel(), distances.ravel(), pixel_magnitudes, normalized_pixel_traces
    
def extract_refactors(codebook: np.ndarray, decoded: np.ndarray, pixel_magnitudes: np.ndarray, normalized_pixel_traces: np.ndarray, extract_backgrounds: bool=False):
    """
    Calculate the scale factors that would result in the mean on bit intensity for each bit to be equal
    """
    n_bc, n_bits = codebook.shape
    if extract_backgrounds:
        background_refactors = extract_backgrounds(decoded, pixel_magnitudes, normalized_pixel_traces)
    else:
        background_refactors = np.zeros(n_bits)
        
    sum_pixel_traces = np.zeros_like(codebook) # compute the summed pixels for each potential barcode in the codebook
    barcodes_seen = np.zeros((n_bc,))
    for b in range(n_bc):
        bc_idx = np.argwhere(decoded == b).ravel()
        barcodes_seen[b] = np.sum(decoded == b)
        for bc in bc_idx:
            norm_pixel_trace = normalized_pixel_traces[bc,:]/np.linalg.norm(normalized_pixel_traces[bc,:])
            sum_pixel_traces[b,:] += norm_pixel_trace/barcodes_seen[b]
    sum_pixel_traces[codebook == 0] = np.nan
    on_bit_intensity = np.nanmean(sum_pixel_traces, axis=0)
    refactors = on_bit_intensity/np.mean(on_bit_intensity)
    return refactors, barcodes_seen

def extract_backgrounds(codebook: np.ndarray, decoded: np.ndarray, pixel_magnitudes: np.ndarray, normalized_pixel_traces: np.ndarray) -> np.ndarray:
    """
    Calculate backgrounds to be subtracted off for the mean off bit intensity for each bit to be equal to zero.
    """
    n_bc, n_bits = codebook.shape
    sum_min_pixel_traces = np.zeros((n_bc, n_bits))
    barcodes_seen = np.zeros(n_bc)
    for b in range(n_bc):
        bc_idx = np.argwhere(decoded == b).ravel()
        barcodes_seen[b] = np.sum(decoded == b)
        for bc in bc_idx:
            sum_min_pixel_traces[b,:] += normalized_pixel_traces[bc,:]
    off_pixel_traces = sum_min_pixel_traces.copy()
    off_pixel_traces[codebook > 0] = np.nan
    off_bit_intensity = np.nansum(off_pixel_traces, axis=0)/np.sum((codebook == 0) * barcodes_seen[:, np.newaxis], axis=0)
    return off_bit_intensity

# XXX Not used currently, based on old merlin function
def get_initial_scale_factors(codebook: np.ndarray, pixel_traces: np.ndarray, n_bins: int=100):
    n_bc, n_bits = codebook.shape
    initial_scale_factors = np.zeros(n_bits)
    for i in range(n_bits):
        h = np.histogram(pixel_traces[:,i],n_bins)
        h = h[1:]
        cumulative_histogram = np.cumsum(h)
        cumulative_histogram = cumulative_histogram/cumulative_histogram[-1]
        initial_scale_factors[i] = np.argmin(np.abs(cumulative_histogram-0.9))+2
    return initial_scale_factors
