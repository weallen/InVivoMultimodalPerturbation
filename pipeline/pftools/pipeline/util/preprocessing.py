from typing import List,Tuple,Optional
import numpy as np
import xarray as xr
import dask.array as da
from abc import ABCMeta
import dask.distributed as dd
from scipy.ndimage import gaussian_filter
from skimage.exposure import equalize_adapthist
from pftools.pipeline.util.filters import high_pass_filter, low_pass_filter
from pftools.pipeline.util.utils import scatter_gather
from pftools.pipeline.util.lrdeconv import deconvolve_lucyrichardson, deconvolve_lucyrichardson_guo
from pftools.pipeline.core.codebook import Codebook
from pftools.pipeline.core.imagestack import imagestack_transform

def adaptive_histogram_equalization(im: np.ndarray, clip_limit:float=0.01, kernel_size:int=256, nbins:int=256) -> np.ndarray:
    """
    Apply adaptive histogram equalization to the image.
    """
    return equalize_adapthist(im / im.max(), clip_limit=clip_limit, kernel_size=kernel_size, nbins=nbins) 

def imagestack_adaptive_histogram_equalization(imstack: da.Array, clip_limit:float=0.01, kernel_size:int=256, nbins:int=256, in_place:bool=False) -> da.Array:
    """
    Apply adaptive histogram equalization to the image stack.
    """
    return imagestack_transform(imstack, adaptive_histogram_equalization, in_place=in_place, clip_limit=clip_limit, kernel_size=kernel_size, nbins=nbins)

def imagestack_clip_high_intensity(imstack: da.Array, z_score_thresh:float=10, in_place:bool=False) -> da.Array:
    """
    Clip high intensity values in the image stack.
    """
    mean = imstack.mean()
    std = imstack.std()
    thresh = mean + z_score_thresh * std
    imstack = imstack.clip(0, thresh)
    return imstack

def imagestack_transpose(imstack: da.Array, in_place:bool=False) -> da.Array:
    """
    Transpose the image stack. 
    """
    return imagestack_transform(imstack, np.transpose, in_place=in_place)

def imagestack_flipud(imstack: da.Array, in_place:bool=False) -> da.Array:
    """
    Flip the image stack along the y axis.
    """
    return imagestack_transform(imstack, np.flipud, in_place=in_place)

def imagestack_fliplr(imstack: da.Array, in_place:bool=False) -> da.Array:
    """
    Flip the image stack along the x axis.
    """
    return imagestack_transform(imstack, np.fliplr, in_place=in_place)

def imagestack_gaussfilt2d(imstack: da.Array, sigma:float, in_place:bool=False) -> da.Array:
    """
    Apply a gaussian filter to the image stack.
    """
    return imagestack_transform(imstack, gaussian_filter, sigma=sigma, in_place=in_place)

def imagestack_lowpass_filter(imstack: da.Array, low_pass_sigma:float, filter_size:Optional[int]=None, in_place:bool=False) -> da.Array:
    """
    Apply a low pass filter to the image stack.
    """
    imstack = imstack.astype(np.float32)
    if filter_size is None:
        filter_size = int(2 * np.ceil(2 * low_pass_sigma) + 1)
    return imagestack_transform(imstack, low_pass_filter, sigma=low_pass_sigma, windowSize=filter_size, in_place=in_place)

def imagestack_highpass_filter(imstack: da.Array, high_pass_sigma:float, filter_size:Optional[int]=None, in_place:bool=False) -> da.Array:
    """
    Apply a high pass filter to the image stack.
    """
    imstack = imstack.astype(np.float32)
    if filter_size is None:
        filter_size = int(2 * np.ceil(2 * high_pass_sigma) + 1) 
    return imagestack_transform(imstack, high_pass_filter, windowSize=filter_size, 
                                sigma=high_pass_sigma, in_place=in_place)

def imagestack_deconvolve(imstack: da.Array, decon_sigma:int=2, 
                     decon_filter_size:Optional[int]=None, decon_iterations:int=20, 
                     in_place:bool=True) -> da.Array:
    """
    Deconvolve the image stack.
    Usually the data is already registered.
    """
    imstack = imstack.astype(np.float32)
    if decon_filter_size is None:
        decon_filter_size = int(2 * np.ceil(2 * decon_sigma) + 1)
    #imstack = imagestack_highpass_filter(imstack, high_pass_sigma, in_place=in_place)
    imstack = imagestack_transform(imstack, deconvolve_lucyrichardson, windowSize=decon_filter_size, sigmaG=decon_sigma, iterationCount=decon_iterations, in_place=in_place)
    #imstack = imstack.astype(np.uint16)
    return imstack

def imagestack_compute_pixel_histogram(imstack: da.Array) -> da.Array:
    """
    Computes a pixel histogram over an image stack of size Nz x Nbit x Nx x Ny 
    """
    n_z, n_bits, _, _ = imstack.shape
    histogram_bins = np.arange(0, np.iinfo(np.uint16).max, 1)
    pixel_histogram = da.zeros((n_bits, len(histogram_bins)-1))
    for z in range(n_z):
        curr_slice = imstack[z]
        for i in range(n_bits):
            pixel_histogram[i,:] += da.histogram(curr_slice[i], bins=histogram_bins)[0]
    return pixel_histogram

def imagestack_scale_readout_images(imstack: da.Array, scale_factors:da.Array) -> da.Array:
    """
    Correct the intensity difference between color channels using existing scale factor profiles
    Basically, median scales each readout channel.
    imstack is Nz x Nbit x Nx x Ny
    """
    if scale_factors is None:
        scale_factors = imagestack_estimate_scale_factors(imstack)

    for i in range(imstack.shape[1]):
        imstack[:,i, :, :] = imstack[:, i, :, :] / scale_factors[i]
    return imstack

def imagestack_estimate_scale_factors(imstack: da.Array) -> da.Array:
    """
    Lazily compute the median intensity of each readout channel.
    imstack is Nz x Nbit x Nx x Ny
    """
    return da.median(imstack.swapaxes(0,1).reshape((imstack.shape[1], -1)), 1)