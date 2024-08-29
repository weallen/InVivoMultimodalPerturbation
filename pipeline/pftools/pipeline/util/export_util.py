# Functions to help with exporting images of proteins
from typing import Tuple, List, Optional
import skimage
import numpy as np

def get_bounding_box_sizes(segmentation_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Get the bounding box sizes of all the 3D segmented regions.'''
    regions = skimage.measure.regionprops(segmentation_mask)
    
    x_sizes = []
    y_sizes = []
    z_sizes = []
    for i in range(len(regions)):
        z0, x0, y0, z1, x1, y1  = regions[i].bbox
        x_sizes.append(x1 - x0)
        y_sizes.append(y1 - y0)
        z_sizes.append(z1 - z0)

    return x_sizes, y_sizes, z_sizes


def select_bounding_boxes(segmentation_mask:np.ndarray, exclude_margin:int=0) -> List[Tuple[Tuple[int, int, int, int], int] ]:
    '''Select bounding boxes from the 2D segmentation mask.
    Args:
        segmentation_mask: The 2D mask of segmentation. Background pixels should have 0 value.
        exclude_margin: The margin size. Any bounding boxes (before regularization) that overlap
            with the margin will be excluded.
    '''
    regions = skimage.measure.regionprops(segmentation_mask)
    x_max = segmentation_mask.shape[0]
    y_max = segmentation_mask.shape[1]
    
    selected_bbs = []
    for i in range(len(regions)):
        x0, y0, x1, y1  = regions[i].bbox
    
        # Exclude the marginal regions
        if ((x0 < exclude_margin) or (y0 < exclude_margin) 
            or (x1 > x_max - exclude_margin) or (y1 > y_max - exclude_margin)):
            continue
            
        selected_bbs.append(((x0, y0, x1, y1), regions[i].label))
         
    return selected_bbs
    
def crop_region(img: np.ndarray, segmentation_mask: np.ndarray, bbox: Tuple[int, int, int int], crop_size:int, region_label:int):
    '''Crop a region out of an image.
    Args:
        img: The image to crop from.
        segmentation_mask: The segmentation mask.
        bbox: The bounding box defined as (x0, y0, x1, y1)
        crop_size: The size of the image that got cropped out.
        region_label: The label of the region in the segmentation mask.
    '''
    x0, y0, x1, y1 = bbox
    bbox_img = img[x0:x1, y0:y1] * (segmentation_mask[x0:x1, y0:y1] == region_label)
    
    # Trim the bounding box image
    if bbox_img.shape[0] > crop_size:
        total_trim = bbox_img.shape[0] - crop_size
        low_trim = int(total_trim / 2)
        bbox_img = bbox_img[low_trim:low_trim + crop_size]
        
    if bbox_img.shape[1] > crop_size:
        total_trim = bbox_img.shape[1] - crop_size
        low_trim = int(total_trim / 2)
        bbox_img = bbox_img[:, low_trim:low_trim + crop_size]
        
    # Get the cropped image with defined size
    x_start = int((crop_size - bbox_img.shape[0]) / 2)
    y_start = int((crop_size - bbox_img.shape[1]) / 2)    
    
    cropped_img = np.zeros((crop_size, crop_size))
    cropped_img[x_start : x_start + bbox_img.shape[0], 
                y_start : y_start + bbox_img.shape[1]] = bbox_img
    
    return cropped_img
