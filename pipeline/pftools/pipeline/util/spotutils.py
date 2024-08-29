
import numpy as np
from typing import Optional, Tuple, List
from dask.delayed import delayed
import dask.array as da
from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors
from pftools.pipeline.spot_tools.Fitting_v4 import get_seed_points_base_v2
from pftools.pipeline.spot_tools.Fitting_v4 import fast_fit_big_image
def filter_neighbors_3d_singleround(data:np.ndarray, radius:float=1, scale=[1.5, 0.11, 0.11]) -> np.ndarray:
    """
    Greedily remove points that are too close to each other in 3D.
    """
    # scale spot positions by the pixel size
    data_scaled = data * np.array(scale)

    # find nearest neighbors in 3D for each spot
    nbs = NearestNeighbors(radius=radius).fit(data_scaled)
    dist, idx = nbs.radius_neighbors(data_scaled)

    # keep track of which points are still valid
    still_valid_idx = np.arange(data.shape[0])

    # iterate through the points and identify those that are too close to each other
    for i in range(data.shape[0]):
        # if the spot has neighbors, remove it's neighbors but keep it       
        if len(idx[i]) > 1:
            still_valid_idx = [j for j in still_valid_idx if j not in idx[i][1:]]


    return data[still_valid_idx]
        
def find_spots_with_thresh(im:np.ndarray, filt_size:int=2, zscore_thresh:float=0.75) -> np.ndarray:
    seeds, _ = get_seed_points_base_v2(im, filt_size=2)
    # filter based on intensity distribution -- needs 2
    seed_inten_zscore = zscore(seeds[3,:])
    seeds = seeds[:,seed_inten_zscore>zscore_thresh]
    seeds = seeds[:3,:].T
    return seeds


def find_spots_across_rounds(imstack:np.ndarray, filt_size:int=3, gfilt_size:int=5, th_seed:float=3., use_fitting:bool=False, zscore_thresh:Optional[float]=None) -> np.ndarray:
    """
    Find seeds in an Z x Round x X x Y stack.
    Returns a (N x 5) array of [Z x Round x X x Y x H] coordinates.
    """
    # seeds are (N x 4) [Z x Y x Y x H]
    #seeds = da.compute([delayed(find_spots_single_image)(imstack[:,i,:,:], 
    #    filt_size=filt_size, gfilt_size=gfilt_size, 
    #    use_fitting=use_fitting, th_seed=th_seed, zscore_thresh=zscore_thresh) 
    #    for i in range(imstack.shape[1])])[0]
    seeds = [find_spots_single_image(imstack[:,i,:,:], 
        filt_size=filt_size, gfilt_size=gfilt_size, 
        use_fitting=use_fitting, th_seed=th_seed, zscore_thresh=zscore_thresh) 
        for i in range(imstack.shape[1])]

    #seeds = [s for s in seeds] # make Nspots x 4
    # set channel value
    for i in range(len(seeds)):
        seeds[i] = np.insert(seeds[i], 1, i, axis=1)
    return np.vstack(seeds)

def find_spots_single_image(img:np.ndarray, filt_size:int=10, gfilt_size:int=2, use_fitting:bool=False, th_seed:float=3., zscore_thresh:Optional[float]=None) -> np.ndarray:
        centers_zxyh, _ = get_seed_points_base_v2(img, filt_size=filt_size, gfilt_size=gfilt_size, th_seed=th_seed)
        if zscore_thresh is not None:
            centers_zxyh = centers_zxyh[:,zscore(centers_zxyh[3,:])>zscore_thresh]
        if use_fitting:
            pts = fast_fit_big_image(img, centers_zxyh[:3,:].T, verbose=False, better_fit=True)
            pts = pts[:,:4] # hf, xc, yc, zc
            pts = pts[pts[:,0]>=0,:] # filter out spots that the gaussian fit thinks have negative indices
            pts = np.swapaxes(pts, 0, 1) # swap to zxyh
            return pts
        else:
            return centers_zxyh.T

def filter_neighbors_2d(pts:np.ndarray, radius:float=5) -> np.ndarray:
    """
    Find mutual nearest neighbors in a 2D plane. 
    data is nRound x X x Y
    """
    added_points = []
    potential_pts = pts.copy()
    unique_rounds = np.unique(pts[:,0])

    updated_pts = pts.copy()
    for r in unique_rounds:
        curr_round_pts = updated_pts[updated_pts[:,0]==r]
        if curr_round_pts.shape[0] > 0:
            nbs = NearestNeighbors(radius=5).fit(updated_pts[:,1:])
            # add points from current round to final set
            for i in range(curr_round_pts.shape[0]):
                added_points.append(curr_round_pts[i][1:])

            # identify neighbors to these points in all rounds, and remove them
            dist, idx = nbs.radius_neighbors(curr_round_pts[:,1:])
            nbor_idx = np.concatenate([i for i in idx])
            still_valid_idx = [i for i in np.arange(updated_pts.shape[0]) if i not in nbor_idx]
            updated_pts = updated_pts[still_valid_idx,:]
    added_points = np.array(added_points)
    return added_points

def filter_neighbors_3d(data:np.ndarray, radius:float=5) -> np.ndarray:
    z_pos = np.unique(data[:,0])
    data_filt = [filter_neighbors_2d(data[data[:,0]==i][:,1:], radius=radius) for i in z_pos]
    for i in range(len(data_filt)):
        data_filt[i] = np.insert(data_filt[i], 0, z_pos[i], axis=1)
    return np.vstack(data_filt)


def in_dim_2d(x,y,xmax,ymax):
    keep = ((x>=0)&(x<xmax)&(y>=0)&(y<ymax))
    return x[keep],y[keep]

def extract_pixel_traces_for_spots(imgs:np.ndarray, pos:np.ndarray, win_size:int=5, bg_win_size:int=15, 
                         subtract_bg:bool=True, do_zscore:bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    imgs is n_round x n_z x n_x x n_y image array 
    """
    #imgs = imgs.max(1)
    n_round, n_z, n_x, n_y = imgs.shape
    n_spots = pos.shape[0]
    bc = np.zeros((n_spots, n_round))
    
    # generate indices of area to keep
    xb,yb = np.reshape(np.indices([win_size*2]*2)-win_size,[2,-1])
    keep = xb*xb + yb*yb <= win_size**2
    xb,yb = xb[keep], yb[keep]
    
    xbg,ybg = np.reshape(np.indices([bg_win_size*2]*2)-bg_win_size,[2,-1])
    keep = (xbg*xbg + ybg*ybg <= bg_win_size**2) & (xbg*xbg + ybg*ybg > win_size**2)
    xbg,ybg = xbg[keep], ybg[keep]
    
    bc = np.zeros((n_spots, n_round))
    bg = np.zeros_like(bc)
    bg_std = np.zeros_like(bc)
    for ic, (zc, xc,yc) in enumerate(pos):
        curr_slice = imgs[:,int(zc),:,:]
        x_keep, y_keep = int(xc)+xb, int(yc)+yb
        x_keep_bg, y_keep_bg = int(xc)+xbg, int(yc)+ybg
                                   
        x_keep,y_keep = in_dim_2d(x_keep, y_keep, n_x, n_y)
        x_keep_bg,y_keep_bg = in_dim_2d(x_keep_bg, y_keep_bg, n_x, n_y)
        for i in range(n_round):
            bc[ic,i] = curr_slice[i][x_keep,y_keep].mean()
            bg[ic,i] = curr_slice[i][x_keep_bg,y_keep_bg].mean()
            bg_std[ic,i] = np.std(curr_slice[i][x_keep_bg,y_keep_bg])
            if subtract_bg:
                bc[ic,i] -= bg[ic,i]
        # normalize each column separately
    if do_zscore:
        for i in range(bc.shape[1]):
            bc[:,i] = zscore(bc[:,i])# bc[:,i].mean()
    return bc, bg, bg_std

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
