from cellpose import models
import os

import numpy as np
from typing import Tuple, Optional, List
import dask.array as da

from tqdm import tqdm
from pftools.pipeline.core.experiment import BaseExperiment
from pftools.pipeline.util.preprocessing import imagestack_flipud, imagestack_fliplr, imagestack_transpose
from dask_image.imread import imread
from pftools.pipeline.util.preprocessing import imagestack_clip_high_intensity, imagestack_adaptive_histogram_equalization
import skimage

def cellpose_segment(im: np.ndarray, channels:Tuple[int,int]=[0,0], diameter:float=160., min_size:float=200., flow_threshold:float=0.4,  
                     model_type:str='cyto3', use_gpu:bool=False, pretrained_model_path:Optional[str]=None) -> np.ndarray:
    """
    Segment cells in the image stack.
    """

    # make segmentation deterministic
    np.random.seed(42)
    if pretrained_model_path is not None:
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained_model_path)
        masks, flows, styles = model.eval(im, diameter=diameter, flow_threshold=flow_threshold, channels=channels, min_size=min_size, do_3D=False, resample=True)
    else:
        model = models.Cellpose(gpu=use_gpu, model_type=model_type)
        masks, flows, styles, diams = model.eval(im, diameter=diameter, flow_threshold=flow_threshold, channels=channels, min_size=min_size, do_3D=False, resample=True)
    return masks # type: ignore


def preprocess_stack_for_segmentation(im: da.Array) -> np.ndarray:
    """
    Preprocess the image stack for segmentation.
    """
    im = imagestack_clip_high_intensity(im, z_score_thresh=10)
    im = imagestack_adaptive_histogram_equalization(im)
    return im     

def get_overlapping_objects(segmentationZ0: np.ndarray,
                            segmentationZ1: np.ndarray,
                            n0: int,
                            fraction_threshold0: float=0.2,
                            fraction_threshold1: float=0.2):
    """compare cell labels in adjacent image masks
    Args:
        segmentationZ0: a 2 dimensional numpy array containing a
            segmentation mask in position Z
        segmentationZ1: a 2 dimensional numpy array containing a
            segmentation mask adjacent to segmentationZ0
        n0: an integer with the index of the object (cell/nuclei)
            to be compared between the provided segmentation masks
    Returns:
        a tuple (n1, f0, f1) containing the label of the cell in Z1
        overlapping n0 (n1), the fraction of n0 overlaping n1 (f0) and
        the fraction of n1 overlapping n0 (f1)
    """

    z1Indexes = np.unique(segmentationZ1[segmentationZ0 == n0])

    z1Indexes = z1Indexes[z1Indexes > 0]

    if z1Indexes.shape[0] > 0:

        # calculate overlap fraction
        n0Area = np.count_nonzero(segmentationZ0 == n0)
        n1Area = np.zeros(len(z1Indexes))
        overlapArea = np.zeros(len(z1Indexes))

        for ii in range(len(z1Indexes)):
            n1 = z1Indexes[ii]
            n1Area[ii] = np.count_nonzero(segmentationZ1 == n1)
            overlapArea[ii] = np.count_nonzero((segmentationZ0 == n0) *
                                               (segmentationZ1 == n1))

        n0OverlapFraction = np.asarray(overlapArea / n0Area)
        n1OverlapFraction = np.asarray(overlapArea / n1Area)
        index = list(range(len(n0OverlapFraction)))

        # select the nuclei that has the highest fraction in n0 and n1
        r1, r2, indexSorted = zip(*sorted(zip(n0OverlapFraction,
                                              n1OverlapFraction,
                                              index),
                                  key=lambda x:x[0]+x[1],
                                  reverse=True))
              
        if (n0OverlapFraction[indexSorted[0]] > fraction_threshold0 and
                n1OverlapFraction[indexSorted[0]] > fraction_threshold1):
            return (z1Indexes[indexSorted[0]],
                    n0OverlapFraction[indexSorted[0]],
                    n1OverlapFraction[indexSorted[0]])
        else:
            return (False, False, False)
    else:
        return (False, False, False)

def combine_2d_segmentation_masks_into_3d(segmentationOutput:
                                          np.ndarray) -> np.ndarray:
    """Take a 3 dimensional segmentation masks and relabel them so that
    nuclei in adjacent sections have the same label if the area their
    overlap surpases certain threshold
    Args:
        segmentationOutput: a 3 dimensional numpy array containing the
            segmentation masks arranged as (z, x, y).
    Returns:
        ndarray containing a 3 dimensional mask arranged as (z, x, y) of
            relabeled segmented cells
    """

    # Initialize empty array with size as segmentationOutput array
    segmentationCombinedZ = np.zeros(segmentationOutput.shape, dtype=int)

    # copy the mask of the section farthest to the coverslip to start
    segmentationCombinedZ[-1, :, :] = segmentationOutput[-1, :, :]
    
    # starting far from coverslip
    for z in range(segmentationOutput.shape[0]-1, 0, -1):

        # get non-background cell indexes for plane Z
        zIndex = np.unique(segmentationCombinedZ[z, :, :])[
                                np.unique(segmentationCombinedZ[z, :, :]) > 0]

        # get non-background cell indexes for plane Z-1
        zm1Index = np.unique(segmentationOutput[z-1, :, :])[
                                np.unique(segmentationOutput[z-1, :, :]) > 0]
        assigned_zm1Index = []
        
        # compare each cell in z0
        for n0 in zIndex:
            n1, f0, f1 = get_overlapping_objects(segmentationCombinedZ[z, :, :],
                                                 segmentationOutput[z-1, :, :],
                                                 n0)
            if n1:
                segmentationCombinedZ[z-1, :, :][
                    (segmentationOutput[z-1, :, :] == n1)] = n0
                assigned_zm1Index.append(n1)
        
        # keep the un-assigned indices in the Z-1 plane
        unassigned_zm1Index = [i for i in zm1Index if i not in assigned_zm1Index]
        max_current_id = np.max(segmentationCombinedZ[z-1:, :, :])
        for i in range(len(unassigned_zm1Index)):
            unassigned_id = unassigned_zm1Index[i]
            segmentationCombinedZ[z-1, :, :][
                    (segmentationOutput[z-1, :, :] == unassigned_id)] = max_current_id + 1 +i
 
    return segmentationCombinedZ

def load_masks(input_path:str, n_tiles:int) -> List[da.Array]:
    """
    Load the masks from the output path.
    """
    masks = []
    for i in tqdm(range(n_tiles)):
        mask_path = os.path.join(input_path, "fov_{}.tiff".format(i))
        masks.append(imread(mask_path))
    return masks

def handle_mask_flips(masks: List[da.Array], experiment:BaseExperiment) -> List[da.Array]:
    for i in range(len(masks)):
        if experiment.microscope_information.flip_vertical:
                print("Flip vertical")
                masks[i] = imagestack_flipud(masks[i], in_place=False)
        if experiment.microscope_information.flip_horizontal:
            print("Flip horizontal")
            for i in range(len(masks)):
                masks[i] = imagestack_fliplr(masks[i], in_place=False)

        if experiment.microscope_information.transpose:
            print("Transpose")
            for i in range(len(masks)):
                masks[i] = imagestack_transpose(masks[i], in_place=False)
        return masks


