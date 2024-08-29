from __future__ import annotations
from typing import Callable, Union, List, Optional, Tuple
import xarray as xr
import functools
import numpy as np
import dask
import dask.array as da
from copy import deepcopy
import dask.bag as db

# 


def make_zslice_fn_xarray_compatible(func, *args, **kwargs):
    """
    This function is used to wrap a function that is applied to 2D arrays compatible with a slice of a 3D numpy array generated
    from map_blocks in xarray.
    Assumes block is first argument. 
    """
    data = args[-1] # data is appended on as last argument when used as a partial function with map_blocks
    subset_args = args[:-1]
    if data.shape[0] != 0:
        temp = data.to_numpy()
        temp = func(np.squeeze(temp), *subset_args, **kwargs)
        temp = temp[np.newaxis,:,:]
        return xr.DataArray(temp, dims=data.dims, coords=data.coords)
    else:
        return data


def imagestack_apply(data:da.Array, func: Callable, *args, **kwargs):
    """
    Calls a function of each z plane of image stack. 
    The function expects an Nx x Ny array as input and returns an ndarray as an output.
    Return list of delayed objects that can be computed in parallel. 
    """
    #z_stack_bag = db.from_sequence([data[i,:,:] for i in range(data.shape[0])])
    #return z_stack_bag.map(func, *args, **kwargs)
    #return [func(data[i,:,:], *args, **kwargs) for i in range(data.shape[0])]
    return da.compute([dask.delayed(func)(data[i,:,:], *args, **kwargs) for i in range(data.shape[0])])[0]

def imagestack_transform(data: da.Array, func: Callable, *args, in_place=False, **kwargs) -> da.Array:
    """
    Apply a function to the whole stack at once, returning a stack.
    Func should take an xarray as input and return an xarray as output of the same dimensionality as each z plane.
    Each of the xarray will be a 1 x Nx x Ny array.
    """
    # make sure that self._data is a dask array
    def wrapper_func(arr, *args, **kwargs) -> Optional[np.ndarray]:
        # We get blocks of shape (1, X, Y) from dask, but want to pass arrays of shape (X, Y) or (Z, X, Y) to func
        # remove extra dimension
        reshaped = np.squeeze(arr) #arr.reshape(arr.shape[1:])
        result = func(reshaped, *args, **kwargs)
        # We then need to reshape the result back to shape (1, X, Y) or (1, 1, X, Y) for dask
        if len(arr.shape) == 3:
            return np.expand_dims(result, axis=0)
        elif len(arr.shape) == 4:
            # if passing in single tiles of shape (1,1,X,Y) then we need to add back the two dimensions
            if len(result.shape) == 2:
                return np.expand_dims(np.expand_dims(result, axis=0), axis=0)
            elif len(result.shape) == 3: 
                return np.expand_dims(result, axis=0)

    if isinstance(data, np.ndarray):
        data = da.from_array(data)
    
    # handle 2D array case
    if data.ndim == 2:
        nx, ny = data.shape
        data = da.expand_dims(data, axis=0)
        data = da.rechunk(data, chunks=(1, nx, ny))
        result = imagestack_transform(data, func, *args, in_place=in_place, **kwargs)
        result = da.squeeze(result, axis=0)
        return result
    else:
        if not in_place:
            temp = data.copy()
            temp = imagestack_transform(temp, func, *args, in_place=True, **kwargs)
            return temp
        else:
            if hasattr(func, '__name__'):
                name = func.__name__
            else:
                name = 'generic_imagestack_transform'
            return data.map_blocks(wrapper_func, *args, dtype=data.dtype, chunks=data.chunksize, **kwargs, name=name) # type: ignore
        
