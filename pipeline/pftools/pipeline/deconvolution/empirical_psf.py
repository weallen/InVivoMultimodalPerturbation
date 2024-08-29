from dexp.optics.psf.microscope_psf import *
from dexp.optics.psf.standard_psfs import *

from scipy.spatial import KDTree
import numpy as np
from dexp.processing.deconvolution.lr_deconvolution import *
from dexp.processing.deconvolution.admm_deconvolution import *
from pftools.pipeline.spot_tools.fitting import get_seeds, select_sparse_centers
from pftools.pipeline.spot_tools.Fitting_v4 import fast_fit_big_image, iter_fit_seed_points, GaussianFit
from pftools.pipeline.spot_tools.fitting import get_seed_points_base

# code to extract PSFs from real images
def get_cropped_points(img, points, crop_size=[10, 10, 10],reject_noncenter=True):
    points = np.round(points).astype(int)
    crops = []
    for i in range(points.shape[0]):
        z,x,y = points[i,:]
        curr_crop = img[(z-crop_size[0]):(z+crop_size[0]),
                (x-crop_size[1]):(x+crop_size[1]),
                (y-crop_size[2]):(y+crop_size[2])]
        # only use points that are actually cenetered on spot
        if reject_noncenter:
            if curr_crop.shape[0] == 2*crop_size[0] and curr_crop.shape[1] == 2*crop_size[1] and curr_crop.shape[2] == 2*crop_size[2]:
                center_value = curr_crop[int(curr_crop.shape[0]/2), int(curr_crop.shape[1]/2), int(curr_crop.shape[2]/2)]
                if center_value == curr_crop.max():
                    crops.append(curr_crop)
    return crops
 
def filter_neighboring_spots(seeds, min_radius=15):
    seeds = seeds.T
    kdtree = KDTree(seeds[:,:3])
    dists = kdtree.query(seeds[:,:3], k=2)[0][:,1]
    return seeds[dists>min_radius,:].T
 
    
def find_beads_v2(img, th_seed=20, filt_size=5, gfilt_size=5, radius_fit=5, min_dist=5, do_filter=False, use_fast=False):
    seeds = get_seed_points_base(img, th_seed=th_seed, filt_size=filt_size, gfilt_size=gfilt_size)
    print(f"--> Found {seeds.shape[1]} seeds")
    if do_filter:
        seeds = filter_neighboring_spots(seeds, min_radius=min_dist)
        print(f"--> {seeds.shape[1]} seeds after filtering")
    if use_fast:
        spots = fast_fit_big_image(img, seeds[:3].T, radius_fit=radius_fit)
    else:
        ft_obj = iter_fit_seed_points(img, seeds[:3], radius_fit=radius_fit)
        ft_obj.firstfit()
        #ft_obj.repeatfit()
        spots = np.array(ft_obj.ps)
    spots_filt = spots.copy()
    final_spots = spots_filt[:,1:4]
    #curr_crops = get_cropped_points(img, final_spots)
    curr_crops = get_cropped_points(img, seeds[:3].T)
    return curr_crops, seeds, spots_filt

def fit_3d_gaussian(gauss):
    """
    Returns height, center_z, center_x, center_y, width_z, background, width_x, width_y
    """
    zb,xb,yb = np.indices(gauss.shape).reshape([3,-1]).astype(int)
    X = np.array([zb,xb,yb])
    gfit = GaussianFit(gauss[zb,xb,yb], X)
    gfit.fit()
    return gfit.p[:8]

def theoretical_res(lam, na):
    """
    Returns diffraction limited resolution in um
    """
    res_xy = lam/(2*na)
    res_z = 2 * lam / na**2
    return res_xy/1000., res_z/1000.



