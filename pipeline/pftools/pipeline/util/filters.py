# From MERlin
# https://github.com/ZhuangLab/MERlin/blob/be3c994ef8fa97fbd3afb85705d8ddbed12118cf/merlin/util/imagefilters.py

import cv2
import numpy as np

"""
This module contains code for performing filtering operations on images
"""
def low_pass_filter(image: np.ndarray,
                    windowSize: int,
                    sigma: float) -> np.ndarray:
    """
    Args:
        image: the input image to be filtered
        windowSize: the size of the Gaussian kernel to use.
        sigma: the sigma of the Gaussian.

    Returns:
        the low pass filtered image. The returned image is the same type
        as the input image.
    """
    return cv2.GaussianBlur(image,
                            (windowSize, windowSize),
                            sigma,
                            borderType=cv2.BORDER_REPLICATE)

def high_pass_filter(image: np.ndarray,
                     windowSize: int,
                     sigma: float) -> np.ndarray:
    """
    Args:
        image: the input image to be filtered
        windowSize: the size of the Gaussian kernel to use.
        sigma: the sigma of the Gaussian.

    Returns:
        the high pass filtered image. The returned image is the same type
        as the input image.
    """
    lowpass = low_pass_filter(image, windowSize, sigma)
    gauss_highpass = image - lowpass  # type: ignore
    gauss_highpass[gauss_highpass < 0] = 0
    return gauss_highpass