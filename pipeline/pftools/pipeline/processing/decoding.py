from typing import Tuple, Optional
import numpy as np
import dask.array as da
from sklearn.neighbors import NearestNeighbors
from pftools.pipeline.util.decodeutil import calculate_normalized_barcodes
from pftools.pipeline.core.codebook import Codebook
from pftools.pipeline.util.preprocessing import imagestack_lowpass_filter
import pandas as pd

def decode_pixels_singleplane_fast(images:np.ndarray,
                              decoding_matrix:np.ndarray,
                              scale_factors:Optional[np.ndarray]=None,
                              backgrounds:Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # decode a single plane image (Nbit x Nx nx Y) fast, no filtering
    if scale_factors is None:
        scale_factors = np.ones(decoding_matrix.shape[1])
    else:
        scale_factors = scale_factors.copy()
    if backgrounds is None:
        backgrounds = np.zeros(decoding_matrix.shape[1])
    else:
        backgrounds = backgrounds.copy()
    n_bit, n_x, n_y = images.shape
    pixel_traces = images.reshape((n_bit, n_x*n_y)).astype(np.float32)
    scaled_pixel_traces = np.array([(p-b)/s for p, s, b in zip(pixel_traces, scale_factors, backgrounds)])
    pixel_magnitudes = np.linalg.norm(scaled_pixel_traces, axis=0)
    pixel_magnitudes[pixel_magnitudes == 0] = 1
    normalized_pixel_traces = scaled_pixel_traces/pixel_magnitudes
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=1).fit(decoding_matrix)
    dist, decoded_image = neighbors.kneighbors(normalized_pixel_traces.T, return_distance=True)
    #decoded_image[dist > distance_threshold] = -1
    #decoded_image[pixel_magnitudes < magnitude_threshold] = -1
    return decoded_image.reshape((n_x, n_y)).astype(np.uint16), pixel_magnitudes.reshape((n_x, n_y)), normalized_pixel_traces.reshape((n_bit, n_x, n_y)), dist.reshape((n_x, n_y))

def imagestack_decode_pixels(filtered_images:np.ndarray,
                decoding_matrix: np.ndarray,
                scale_factors:Optional[np.ndarray]=None,
                backgrounds:Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # ImageStack is an unrolled Nz x Nbit x Nx x Ny matrix (i.e. a Z-slice)
    # each pixel is processed independently, so can be chunked and operated on as
    # subpieces of a larger numpy array

    if scale_factors is None:
        scale_factors = np.ones(decoding_matrix.shape[1])
    else:
        scale_factors = scale_factors.copy()
    if backgrounds is None:
        backgrounds = np.zeros(decoding_matrix.shape[1])
    else:
        backgrounds = backgrounds.copy()


    n_z, n_bit, n_x, n_y = filtered_images.shape

    n_pix = n_x * n_y
    normalized_pixel_traces = np.reshape(filtered_images.copy(), (n_z, n_bit, n_pix)).astype(np.float32)
    for z in range(n_z):
        normalized_pixel_traces[z] = np.array([(p-b)/s for p,s,b in zip(normalized_pixel_traces[z], scale_factors, backgrounds)])

    pixel_magnitudes = np.zeros((n_z, n_pix), dtype=np.float32)
    for z in range(n_z):
        temp = np.linalg.norm(normalized_pixel_traces[z], axis=0)
        temp[temp == 0] = 1
        pixel_magnitudes[z] = temp

    # divide
    normalized_pixel_traces = normalized_pixel_traces/pixel_magnitudes[:, None, :]

    # chunk up the images automatically, where each chunk gets all n_bit values
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=1).fit(decoding_matrix)
    decoded_image = np.zeros((n_z, n_pix)).astype(np.uint16)
    distances = np.zeros((n_z, n_pix)).astype(np.float32)
    for z in range(n_z):
        dist, idx = neighbors.kneighbors(normalized_pixel_traces[z].T, return_distance=True)
        decoded_image[z] = np.squeeze(idx)
        distances[z] = np.squeeze(dist)

    # get rid of pixels below the thresholds
    #decoded_image[distances > distance_threshold] = -1
    #decoded_image[pixel_magnitudes < magnitude_threshold] = -1
    pixel_magnitudes = np.reshape(pixel_magnitudes, (n_z, n_x, n_y))
    normalized_pixel_traces = np.reshape(normalized_pixel_traces, (n_z, n_bit, n_x, n_y))
    distances = np.reshape(distances, (n_z, n_x, n_y))
    decoded_image = np.reshape(decoded_image, distances.shape)
    return decoded_image, pixel_magnitudes, normalized_pixel_traces, distances


def imagestack_decode_pixels_dask(filtered_images:da.Array,
                decoding_matrix: np.ndarray,
                scale_factors:Optional[np.ndarray]=None,
                backgrounds:Optional[np.ndarray]=None,
                distance_threshold:float=0.5176,
                magnitude_threshold:float=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Don't use this -- probably much slower, and has memory issues
    # ImageStack is an unrolled Nz x Nbit x Nx x Ny matrix (i.e. a Z-slice)
    # each pixel is processed independently, so can be chunked and operated on as
    # subpieces of a larger numpy array

    def _compute_scaled_pixel_traces(pt:np.ndarray) -> np.ndarray:
        """
        Pixel traces will be (1 x Nbit x Nx*Ny)  chunk
        """
        pt = pt.reshape(pt.shape[1:])
        spt = np.array([(p-b)/s for p, s, b in zip(pt, scale_factors, backgrounds)]) # type: ignore
        return np.expand_dims(spt, 0)

    def _compute_pixel_magnitudes(spt:np.ndarray) -> np.ndarray:
        temp = np.linalg.norm(spt.reshape(spt.shape[1:]), axis=0)
        temp[temp == 0] = 1
        return np.expand_dims(np.expand_dims(temp,0),0)

    def _decode_chunk(npt:np.ndarray) -> np.ndarray:
        """
        Takes in an (1 x Nbit x Npix) chunk, where Npix is the number of pixels in the chunk, determined automatically by dask.
        """
        npt = npt.reshape(npt.shape[1:]).T
        dist, idx = neighbors.kneighbors(npt, return_distance=True)
        # return a (1,2,Nx*Ny) matrix
        return np.expand_dims(np.stack((np.squeeze(idx), np.squeeze(dist))),0)

    if scale_factors is None:
        scale_factors = np.ones(decoding_matrix.shape[1])
    else:
        scale_factors = scale_factors.copy()
    if backgrounds is None:
        backgrounds = np.zeros(decoding_matrix.shape[1])
    else:
        backgrounds = backgrounds.copy()


    n_z, n_bit, n_x, n_y = filtered_images.shape
    #if low_pass_sigma > 0:
        # low pass filter image -- blur the deconvolved spots
    #    filtered_images = filtered_images.rechunk((1, 1, n_x, n_y))
    #    filtered_images = imagestack_lowpass_filter(filtered_images, low_pass_sigma, in_place=True)

    # turn into an Nz x Nround x Nx*Ny for decoding
    # chunk along the z axis for paralleliation
    normalized_pixel_traces = da.reshape(filtered_images, (n_z, n_bit, n_x*n_y)).rechunk(chunks=(1, int(n_bit), int(n_x*n_y))) # type: ignore
    n_pix = n_x * n_y

    # run compute on each of these so don't redo it later
    normalized_pixel_traces = normalized_pixel_traces.map_blocks(_compute_scaled_pixel_traces, dtype=np.float32, chunks=(1, n_bit, n_pix), name='compute_scaled_pixel_traces') # type: ignore
    pixel_magnitudes = normalized_pixel_traces.map_blocks(_compute_pixel_magnitudes, dtype=np.float32, chunks=(1, 1, n_pix), name='compute_pixel_magnitudes')
    pixel_magnitudes = da.reshape(pixel_magnitudes, (n_z, n_x*n_y)) # remove the extra dimension

    # divide
    normalized_pixel_traces = normalized_pixel_traces/pixel_magnitudes[:, None, :]

    # chunk up the images automatically, where each chunk gets all n_bit values
    #normalized_pixel_traces = normalized_pixel_traces
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=1).fit(decoding_matrix)
    output = normalized_pixel_traces.map_blocks(_decode_chunk,
                                                chunks=(1, 2, normalized_pixel_traces.chunksize[2]),
                                                dtype=np.float64,
                                                drop_axis=1, new_axis=[1], name='decode_chunk')
    decoded_image = output[:, 0, :].astype(np.int16)
    distances = output[:, 1, :].astype(np.float32)

    # get rid of pixels below the thresholds
    decoded_image[distances > distance_threshold] = -1
    decoded_image[pixel_magnitudes < magnitude_threshold] = -1
    pixel_magnitudes = da.reshape(pixel_magnitudes, (n_z, n_x, n_y))
    normalized_pixel_traces = da.reshape(normalized_pixel_traces, (n_z, n_bit, n_x, n_y))
    distances = da.reshape(distances, (n_z, n_x, n_y))
    decoded_image = da.reshape(decoded_image, distances.shape)
    return da.compute(decoded_image, pixel_magnitudes, normalized_pixel_traces, distances) # type: ignore

