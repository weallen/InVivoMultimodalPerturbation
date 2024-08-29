from typing import List, Tuple, Optional
import numpy as np
import functools
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import dask
import dask.array as da
import xarray as xr
from pftools.pipeline.core.imagestack import imagestack_transform, imagestack_apply
from pftools.pipeline.util.preprocessing import imagestack_highpass_filter
from pftools.pipeline.util.filters import high_pass_filter
def register_translation_2d(
    moving: np.ndarray,
    reference: np.ndarray,
    upscale:int=16) -> np.ndarray:
    #print(np.array(reference).shape, np.array(moving).shape)
    s, _, _ = phase_cross_correlation(np.array(reference), np.array(moving), upsample_factor=upscale)
    return s

def imagestack_compute_registration_to_reference(stack:da.Array, reference:da.Array, upscale:int=16) -> List[np.ndarray]:
    ref = reference.compute()
    stack = stack.compute()
    #print(ref.shape, stack.shape)
    reg_fn = functools.partial(register_translation_2d, reference=ref, upscale=upscale)
    shifts = imagestack_apply(stack, reg_fn)
    return shifts

def imagestack_fiducial_correlation_warp_to_reference(fiducial_stack: da.Array, reference:da.Array, return_reg=False, mode:str='mirror', high_pass_sigma:int=3, upscale:int=100, in_place:bool=False) -> Tuple[Optional[xr.DataArray], List[np.ndarray]]:
    """
    Register all images to first image in stack, first performing a high pass filter.
    """
    filtered_imgs = imagestack_highpass_filter(fiducial_stack, high_pass_sigma, in_place=in_place)
    filtered_reference = imagestack_highpass_filter(reference, high_pass_sigma, in_place=in_place)
    shifts = imagestack_compute_registration_to_reference(filtered_imgs, filtered_reference, upscale=upscale)
    if return_reg:
        shifted_fiducial = imagestack_apply_shift(fiducial_stack, shifts, mode=mode)
        return shifted_fiducial, shifts
    else:
        return None, shifts

def imagestack_compute_registration(stack:da.Array, upscale:int=16, reference_idx:int=0) -> List[np.ndarray]:
    """
    Register all images to first image in stack.
    Assumes dimensions Nimg x Nx x Ny
    shifts is of length Nimg
    """
    # get the image_idx of each image
    reference = stack[reference_idx].compute()
    reg_fn = functools.partial(register_translation_2d, reference=reference, upscale=upscale)
    shifts = imagestack_apply(stack.compute(), reg_fn)
    # XXX this was only for testing -- turn off or it will break everything
    #shifts = [i*np.random.random()*1000 for i in shifts]
    return shifts

def imagestack_apply_shift(stack: da.Array, shifts: List[np.ndarray], mode: str='constant') -> da.Array:
    """
    Shift all images in stack by the specified shifts.
    Shifts are 2x1 arrays.
    Assumes dimensions Nimg x Nx x Ny
    """
    transformed_stack = stack.copy()
    # split up the data by image_idx and apply the shifts to each
    #shifted_stack = dask.compute([dask.delayed(_apply_shift)(stack.sel_value('image_idx', image_idx[i]), shifts[i]) for i in range(len(shifts))])[0]
    #shifted_stack = concatenate_imagestacks(shifted_stack)
    for i in range(len(shifts)):
        curr_img = transformed_stack[i]
        if curr_img.ndim == 2:
            curr_img = da.expand_dims(curr_img, axis=0)
        curr_img = imagestack_transform(curr_img, functools.partial(shift, shift=shifts[i], mode=mode), in_place=True)
        transformed_stack[i] = da.squeeze(curr_img)
    return transformed_stack

def imagestack_fiducial_correlation_warp(fiducial_stack: da.Array, return_reg=False, reference_idx:int=0, mode:str='mirror', high_pass_sigma:int=3, upscale:int=100, in_place:bool=False) -> Tuple[Optional[xr.DataArray], List[np.ndarray]]:
    """
    Register all images to first image in stack, first performing a high pass filter.
    """
    filtered_imgs = imagestack_highpass_filter(fiducial_stack, high_pass_sigma, in_place=in_place)
    shifts = imagestack_compute_registration(filtered_imgs, upscale=upscale, reference_idx=reference_idx)
    if return_reg:
        shifted_fiducial = imagestack_apply_shift(fiducial_stack, shifts, mode=mode)
        return shifted_fiducial, shifts
    else:
        return None, shifts

def apply_shifts_by_image_index(images: xr.DataArray, shifts: List[np.ndarray], image_idx:List[int]=None, mode: str='mirror',in_place=True) -> xr.DataArray:
    """
    Shift all images in stack by the specified shifts.
    Shifts are 2x1 arrays.
    Assumes dimensions Nimg x Nx x Ny
    """
    if in_place:
        warp_images = images
    else:
        warp_images = images.copy(deep=True)
    if image_idx is None:
        image_idx = images.coords['image_idx'].values
    for i in range(len(shifts)):
        curr_img = warp_images[warp_images.coords['image_idx'] == image_idx[i]]
        if curr_img.ndim == 2:
            curr_img = da.expand_dims(curr_img, axis=0)
        # XXX Should this be squeezed?
        curr_img.data = imagestack_transform(curr_img.data, functools.partial(shift, shift=shifts[i], mode=mode), in_place=True)
        #curr_img.data = imagestack_apply_shift(curr_img.data, #imagestack_transform(curr_img.data, functools.partial(shift, shift=shifts[i], mode=mode), in_place=True)
        warp_images[warp_images.coords['image_idx'] == image_idx[i]] = curr_img
    return warp_images

def register_tile_by_image_index(fiducial: xr.DataArray, images: xr.DataArray, reference_idx: int=0, mode: str='mirror',in_place=False) -> Tuple[xr.DataArray, np.ndarray]:
    """
    Computes warp from fiducials then applies shifts to a DataArray containing a stack of images that are indexed by the coordinate image_idx.
    """
    if in_place:
        warp_images = images
        warp_fiducials = fiducial
    else:
        warp_images = images.copy(deep=True)
        warp_fiducials = fiducial.copy(deep=True)
    image_idx = fiducial.coords['image_idx'].values
    _, shifts = imagestack_fiducial_correlation_warp(fiducial.data, reference_idx=reference_idx, mode=mode, return_reg=False)
    shifted_images = apply_shifts_by_image_index(images, shifts, image_idx=list(fiducial.coords['image_idx'].values))
    return shifted_images, shifts


def register_tile_to_reference(tile: xr.DataArray, fiducial_tile:xr.DataArray, reference:xr.DataArray, return_reg_fiducials=False) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    reg_fiducials, shifts = imagestack_fiducial_correlation_warp_to_reference(fiducial_tile.data, reference.data, return_reg=return_reg_fiducials)
    shifted_tile = apply_shifts_by_image_index(tile, shifts, image_idx=list(fiducial_tile.coords['image_idx'].values))
    if return_reg_fiducials:
        return shifted_tile, shifts, reg_fiducials.compute() # type: ignore
    else:
        return shifted_tile, shifts, None # type: ignore


def register_tile(tile: xr.DataArray, fiducial_tile: xr.DataArray, return_reg_fiducials=False) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """
    Register a single tile to a fiducial tile.
    Returns the dask array for the tile and the shifts.
    """
    reg_fiducials, shifts = imagestack_fiducial_correlation_warp(fiducial_tile.data, return_reg=return_reg_fiducials)
    shifted_tile = apply_shifts_by_image_index(tile, shifts, image_idx=list(fiducial_tile.coords['image_idx'].values))
    if return_reg_fiducials:
        return shifted_tile, shifts, reg_fiducials.compute() # type: ignore
    else:
        return shifted_tile, shifts, None # type: ignore



        
