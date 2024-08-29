# From MERlin: https://raw.githubusercontent.com/ZhuangLab/MERlin/be3c994ef8fa97fbd3afb85705d8ddbed12118cf/merlin/util/decoding.py
import numpy as np
import pandas
import cv2
from typing import Tuple
from typing import Dict
from skimage import measure
import dask.array as da
import pandas as pd
from pftools.pipeline.util import binary
from pftools.pipeline.core.codebook import Codebook


def normalize(x):
    norm = np.linalg.norm(x)
    if norm > 0: 
        return x/norm
    else:
        return x

def calculate_normalized_barcodes(codebook: Codebook, 
                                  ignore_blanks:bool=False,
                                  include_errors:bool=False) -> np.ndarray:
    """Normalize the barcodes present in the provided codebook so that
    their L2 norm is 1.

    Args:
        ignoreBlanks: Flag to set if the barcodes corresponding to blanks
            should be ignored. If True, barcodes corresponding to a name
            that contains 'Blank' are ignored.
        includeErrors: Flag to set if barcodes corresponding to single bit 
            errors should be added.
    Returns:
        A 2d numpy array where each row is a normalized barcode and each
            column is the corresponding normalized bit value.
    """    
    barcodeSet = codebook.get_barcodes(ignoreBlanks=ignore_blanks)
    magnitudes = np.sqrt(np.sum(barcodeSet*barcodeSet, axis=1))
    
    if not include_errors:
        weightedBarcodes = np.array(
            [normalize(x) for x, m in zip(barcodeSet, magnitudes)])
        return weightedBarcodes

    else:
        barcodesWithSingleErrors = []
        for b in barcodeSet:
            barcodeSet = np.array([b]
                                    + [binary.flip_bit(b, i)
                                        for i in range(len(b))])
            bcMagnitudes = np.sqrt(np.sum(barcodeSet*barcodeSet, axis=1))
            weightedBC = np.array(
                [x/m for x, m in zip(barcodeSet, bcMagnitudes)])
            barcodesWithSingleErrors.append(weightedBC)
        return np.array(barcodesWithSingleErrors)

def extract_refactors_from_molecules(mols:pd.DataFrame, decoding_matrix:np.ndarray, extract_backgrounds:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    barcode_count, bit_count = decoding_matrix.shape
    intensity_cols = [i for i in mols.columns if i.startswith('intensity_')]
    intensity_vals = mols.loc[:,intensity_cols].values
    mag_vals = mols.loc[:, 'mean_intensity'].values
    if extract_backgrounds:
        background_refactors = extract_backgrounds_from_molecules(mols, decoding_matrix)
    else:
        background_refactors = np.zeros(intensity_vals.shape[1])
    sum_pixel_traces = np.zeros((barcode_count, bit_count))
    barcodes_seen = np.zeros(barcode_count)
    for b in range(barcode_count):
        barcode_idx = np.argwhere(mols.barcode_id==b).flatten()
        barcodes_seen[b] = len(barcode_idx)
        pixel_trace = (mag_vals[barcode_idx] * intensity_vals[barcode_idx,:].T).T - background_refactors # get the background subtracted raw intensity values for each molecule for each bit
        norm_pixel_trace = (pixel_trace.T/np.linalg.norm(pixel_trace.T, axis=0)).T # normalize the pixel traces
        sum_pixel_traces[b, :] = np.mean(norm_pixel_trace, axis=0) # take the mean of the normalized pixel traces
    sum_pixel_traces[decoding_matrix == 0] = np.nan
    on_bit_intensity = np.nanmean(sum_pixel_traces, axis=0)
    refactors = on_bit_intensity/np.mean(on_bit_intensity)
    return refactors, background_refactors, barcodes_seen

def extract_backgrounds_from_molecules(mols:pd.DataFrame, decoding_matrix:np.ndarray) -> np.ndarray:
    barcode_count, bit_count = decoding_matrix.shape
    intensity_cols = [i for i in mols.columns if i.startswith('intensity_')]
    intensity_vals = mols.loc[:,intensity_cols].values
    mag_vals = mols.loc[:, 'mean_intensity'].values
    sum_min_pixel_traces = np.zeros((barcode_count, bit_count))
    barcodes_seen = np.zeros(barcode_count)
    for b in range(barcode_count):
        barcode_idx = np.argwhere(mols.barcode_id==b).flatten()
        barcodes_seen[b] = len(barcode_idx)
        pixel_trace = (mag_vals[barcode_idx] * intensity_vals[barcode_idx,:].T).T # get the raw intensity values for each molecule for each bit
        sum_min_pixel_traces[b, :] = np.mean(pixel_trace, axis=0) # take the average of the raw intensity values for each bit 
    off_pixel_traces = sum_min_pixel_traces.copy()
    off_pixel_traces[decoding_matrix > 0] = np.nan
    off_bit_intensity = np.nansum(off_pixel_traces, axis=0)/np.sum((decoding_matrix == 0) * barcodes_seen[:, np.newaxis], axis=0)
    return off_bit_intensity


def extract_refactors(decoding_matrix:np.ndarray,
                    decodedImage, pixelMagnitudes, normalizedPixelTraces,
                    extractBackgrounds = False,
                    refactor_area_threshold:float=4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the scale factors that would result in the mean
    on bit intensity for each bit to be equal.

    This code follows the legacy matlab decoder.

    If the scale factors for this decoder are not set to 1, then the
    calculated scale factors are dependent on the input scale factors
    used for the decoding.

    Args:
        imageSet: the image stack to decode in order to determine the
            scale factors
    Returns:
            a tuple containing an array of the scale factors, an array
            of the backgrounds, and an array of the abundance of each
            barcode determined during the decoding. For the scale factors
            and the backgrounds, the i'th entry is the scale factor
            for bit i. If extractBackgrounds is false, the returned
            background array is all zeros.
    """

    barcode_count, bit_count = decoding_matrix.shape

    if extractBackgrounds:
        backgroundRefactors = extract_backgrounds(decoding_matrix,
            decodedImage, pixelMagnitudes, normalizedPixelTraces)
    else:
        backgroundRefactors = np.zeros(bit_count)

    sumPixelTraces = np.zeros((barcode_count, bit_count))
    barcodesSeen = np.zeros(barcode_count)
    for b in range(barcode_count):
        barcodeRegions = [x for x in measure.regionprops(
                    measure.label((decodedImage == b).astype(np.int)))
                            if x.area >= refactor_area_threshold]
        barcodesSeen[b] = len(barcodeRegions)
        for br in barcodeRegions:
            meanPixelTrace = \
                np.mean([normalizedPixelTraces[:, y[0],
                            y[1]]*pixelMagnitudes[y[0], y[1]]
                            for y in br.coords], axis=0) - backgroundRefactors
            normPixelTrace = meanPixelTrace/np.linalg.norm(meanPixelTrace)
            sumPixelTraces[b, :] += normPixelTrace/barcodesSeen[b]

    sumPixelTraces[decoding_matrix == 0] = np.nan
    onBitIntensity = np.nanmean(sumPixelTraces, axis=0)
    refactors = onBitIntensity/np.mean(onBitIntensity)

    return refactors, backgroundRefactors, barcodesSeen

def extract_backgrounds(
        decoding_matrix: np.ndarray, decodedImage: np.ndarray, pixelMagnitudes: np.ndarray, normalizedPixelTraces: np.ndarray
) -> np.ndarray:
    """Calculate the backgrounds to be subtracted for the the mean off
    bit intensity for each bit to be equal to zero.

    Args:
        imageSet: the image stack to decode in order to determine the
            scale factors
    Returns:
        an array of the backgrounds where the i'th entry is the scale factor
            for bit i.
    """
    barcode_count, bit_count = decoding_matrix.shape
    sumMinPixelTraces = np.zeros((barcode_count, bit_count))
    barcodesSeen = np.zeros(barcode_count)
    # TODO this core functionality is very similar to that above. They
    # can be abstracted
    for b in range(barcode_count):
        barcodeRegions = [x for x in measure.regionprops(
            measure.label((decodedImage == b).astype(np.int)))
                            if x.area >= 5]
        barcodesSeen[b] = len(barcodeRegions)
        for br in barcodeRegions:
            minPixelTrace = \
                np.min([normalizedPixelTraces[:, y[0],
                        y[1]] * pixelMagnitudes[y[0], y[1]]
                        for y in br.coords], axis=0)
            sumMinPixelTraces[b, :] += minPixelTrace

    offPixelTraces = sumMinPixelTraces.copy()
    offPixelTraces[decoding_matrix > 0] = np.nan
    offBitIntensity = np.nansum(offPixelTraces, axis=0) / np.sum(
        (decoding_matrix == 0) * barcodesSeen[:, np.newaxis], axis=0)
    backgroundRefactors = offBitIntensity

    return backgroundRefactors



