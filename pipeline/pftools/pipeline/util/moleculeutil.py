# From MERlin: https://raw.githubusercontent.com/ZhuangLab/MERlin/be3c994ef8fa97fbd3afb85705d8ddbed12118cf/merlin/util/barcodefilters.py 
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import pandas as pd
from typing import List, Tuple, Optional
from skimage import measure
from pftools.pipeline.core.codebook import Codebook
import dask.array as da

from scipy import optimize

def assign_gene_symbols_to_mols(mols:pd.DataFrame, codebook:Codebook) -> pd.DataFrame:
    symbols = pd.Categorical([codebook.get_name_for_barcode_index(i) for i in mols.barcode_id])
    mols = mols.assign(gene_symbol = symbols)
    return mols

def crop_molecules(mols:pd.DataFrame, crop_percent:float, fov_size_x:int, fov_size_y:int) -> pd.DataFrame:
    """
    Crop molecules to within some percentage of size of FOV, to avoid overlap betwen tiles
    """
    crop_width_x = int(crop_percent * fov_size_x)
    crop_width_y = int(crop_percent * fov_size_y)
    return mols[(mols['x'].between(crop_width_x, fov_size_x - crop_width_x, inclusive='neither')) & (mols['y'].between(crop_width_y, fov_size_y - crop_width_y, inclusive='neither'))]

def extract_barcodes_by_index(decoded:np.ndarray, distances:np.ndarray, magnitudes:np.ndarray, barcode_indexes:np.ndarray, probabilities:Optional[np.ndarray]=None) -> pd.DataFrame:
    props_b = extract_barcode_regionprops(decoded, decoded, barcode_indexes)
    props_d = extract_barcode_regionprops(decoded, distances, barcode_indexes)
    props_m = extract_barcode_regionprops(decoded, magnitudes, barcode_indexes)
    if probabilities is not None:
        props_p = extract_barcode_regionprops(decoded, probabilities, barcode_indexes)
    columns = ["x", "y", 
            "x_max", "y_max",
            "x_start", "x_end", 
            "y_start", "y_end",
            "barcode_id", 
            "magnitude", 
            "magnitude_min", 
            "magnitude_max", 
            "distance", 
            "distance_min", 
            "distance_max", 
            "area"]
    if probabilities is not None:
        columns.append("likelihood")
    if len(props_d) == 0:
        return pd.DataFrame(columns = columns)
    else:
        bc_ids = np.array([prop.min_intensity for prop in props_b])
        x_start = np.array([prop.bbox[0] for prop in props_b]).astype(np.float32)
        x_end = np.array([prop.bbox[2] for prop in props_b]).astype(np.float32)
        y_start = np.array([prop.bbox[1] for prop in props_b]).astype(np.float32)
        y_end = np.array([prop.bbox[3] for prop in props_b]).astype(np.float32)
        centroid_coords = np.array([prop.weighted_centroid for prop in props_b])
        centroid_coords = centroid_coords[:, [1,0]]
        max_coords = np.array([prop.coords[magnitudes[prop.coords[:,0], prop.coords[:,1]].argmax()] for prop in props_b]).astype(np.float32)
        max_coords = max_coords[:, [1,0]] # swap x and y 
        areas = np.array([prop.area for prop in props_b]).astype(np.float32)
        mags_mean = np.array([prop.mean_intensity for prop in props_m]).astype(np.float32)
        mags_min = np.array([prop.min_intensity for prop in props_m]).astype(np.float32)
        mags_max = np.array([prop.max_intensity for prop in props_m]).astype(np.float32)
        dists_mean = np.array([prop.mean_intensity for prop in props_d]).astype(np.float32)
        dists_min = np.array([prop.min_intensity for prop in props_d]).astype(np.float32)
        dists_max = np.array([prop.max_intensity for prop in props_d]).astype(np.float32)
        if probabilities is not None:
            likelihoods = np.array([-np.sum(np.log10(1-x.intensity_image[x.image])) for x in props_p]).astype(np.float32) # type: ignore
        df = pd.DataFrame(data = {
            "x":centroid_coords[:,0],
            "y":centroid_coords[:,1],
            "x_max":max_coords[:,0],
            "y_max":max_coords[:,1],
            "x_start":x_start,
            "x_end":x_end,
            "y_start":y_start,
            "y_end":y_end,
            "barcode_id":bc_ids,
            "magnitude":mags_mean,
            "magnitude_min":mags_min,
            "magnitude_max":mags_max,
            "distance":dists_mean,
            "distance_min":dists_min,
            "distance_max":dists_max,
            "area":areas})
        if probabilities is not None:
            # add likelihoods column
            df["likelihood"] = likelihoods # type: ignore
        return df                 

def extract_barcode_regionprops(labels:np.ndarray, values:np.ndarray, barcode_indexes:Optional[np.ndarray]=None) -> List[measure._regionprops.RegionProperties]:
    """
    Extract region properties from a labeled image.
    """
    if barcode_indexes is None:
        barcode_indexes = np.unique(labels)
    return measure.regionprops(measure.label(np.isin(labels, barcode_indexes)), intensity_image=values, cache=False)

def calc_barcode_fdr(barcodes:pd.DataFrame, codebook:Codebook) -> float:
    blank_indexes = codebook.get_blank_indexes()
    n_barcode = codebook.get_barcode_count()
    blanks = barcodes[barcodes.barcode_id.isin(blank_indexes)]
    n_blank = blanks.shape[0]
    n_total = barcodes.shape[0] + 1
    fdr = (n_blank/len(blank_indexes))/(n_total / n_barcode)
    return fdr

def estimate_lik_err_table(
    bd: pd.DataFrame, cb: Codebook, minScore:int=0, maxScore:int=10, bins:int=100) -> dict:
    """
    This function estimates the likelihood error table for a given set of barcodes.
    Only works if have likelihood values from Bayesian decoding. 
    """
    scores = np.linspace(minScore, maxScore, bins)
    blnkBarcodeNum = len(cb.get_blank_indexes())
    codeBarcodeNum = len(cb.get_coding_indexes()) + len(cb.get_blank_indexes())
    pvalues = dict()
    for s in scores:
        bd = bd[bd.likelihood >= s]
        numPos = np.count_nonzero(
            bd.barcode_id.isin(cb.get_coding_indexes()))
        numNeg = np.count_nonzero(
            bd.barcode_id.isin(cb.get_blank_indexes()))
        numNegPerBarcode = numNeg / blnkBarcodeNum
        numPosPerBarcode = (numPos + numNeg) / codeBarcodeNum
        pvalues[s] = numNegPerBarcode / numPosPerBarcode
    return pvalues

def estimate_barcode_threshold(barcodes:pd.DataFrame, codebook:Codebook, cutoff:float=0.05, bins=100) -> float:
    tab = estimate_lik_err_table(barcodes, codebook, bins=bins, minScore=0,maxScore=10)
    return min(np.array(list(tab.keys()))[np.array(list(tab.values())) < cutoff])

def filter_molecules(barcodes:pd.DataFrame, codebook:Codebook,
                    keep_blank:bool=False, min_area_size:int=1, likelihood_thresh:Optional[float]=None) -> pd.DataFrame:
    barcodes = barcodes[barcodes.area >= min_area_size]
    if likelihood_thresh is not None:
        barcodes = barcodes[barcodes.likelihood >= likelihood_thresh]
    if not keep_blank:
        barcodes = barcodes[barcodes.barcode_id.isin(codebook.get_coding_indexes())]
    return barcodes



def remove_zplane_duplicates_all_barcodeids(barcodes: pd.DataFrame,
                                            zPlanes: int,
                                            maxDist: float,
                                            allZPos: List) -> pd.DataFrame:
    """ Depending on the separation between z planes, spots from a single
        molecule may be observed in more than one z plane. These putative
        duplicates are removed based on supplied distance and z plane
        constraints. In evaluating this method, when z planes are separated
        by 1.5 µm the likelihood of finding a putative duplicate above or below
        the selected plane is ~5-10%, whereas the false-positive rate is closer
        to 1%, as determined by checking two planes above or below, or comparing
        barcodes of different identities but similar abundance between
        adjacent z planes.

    Args:
        barcodes: a pandas dataframe containing all the entries for a given
                  barcode identity
        zPlanes: number of planes above and below to consider when evaluating
                 potential duplicates
        maxDist: maximum euclidean distance allowed to separate centroids of
                 putative barcode duplicate, in pixels
    Returns:
        keptBarcodes: pandas dataframe where barcodes of the same identity that
                      fall within parameters of z plane duplicates have
                      been removed.
    """
    if len(barcodes) == 0:
        return barcodes
    else:
        barcodeGroups = barcodes.groupby('barcode_id')
        bcToKeep = []
        for bcGroup, bcData in barcodeGroups:
            bcToKeep.append(
                remove_zplane_duplicates_single_barcodeid(bcData, zPlanes,
                                                          maxDist, allZPos))
        mergedBC = pd.concat(bcToKeep, axis=0).reset_index(drop=True)
        mergedBC = mergedBC.sort_values(by=['barcode_id', 'z'])
        return mergedBC


def remove_zplane_duplicates_single_barcodeid(barcodes: pd.DataFrame,
                                              zPlanes: int,
                                              maxDist: float,
                                              allZPos: List) -> pd.DataFrame:
    """ Remove barcodes with a given barcode id that are putative z plane
        duplicates.

    Args:
        barcodes: a pandas dataframe containing all the entries for a given
                  barcode identity
        zPlanes: number of planes above and below to consider when evaluating
                 potential duplicates
        maxDist: maximum euclidean distance allowed to separate centroids of
                 putative barcode duplicate, in pixels
    Returns:
        keptBarcodes: pandas dataframe where barcodes of the same identity that
                      fall within parameters of z plane duplicates have
                      been removed.
    """
    barcodes.reset_index(drop=True, inplace=True)
    if not len(barcodes['barcode_id'].unique()) == 1:
        errorString = 'The method remove_zplane_duplicates_single_barcodeid ' +\
                      'should be given a dataframe containing molecules ' +\
                      'that all have the same barcode id. Please use ' +\
                      'remove_zplane_duplicates_all_barcodeids to handle ' +\
                      'dataframes containing multiple barcode ids'
        raise ValueError(errorString)
    graph = nx.Graph()
    zPos = sorted(allZPos)
    graph.add_nodes_from(barcodes.index.values.tolist())
    for z in range(0, len(zPos)):
        zToCompare = [pos for pos, otherZ in enumerate(zPos) if
                      (pos >= z - zPlanes) & (pos <= z + zPlanes) & ~(pos == z)]
        treeBC = barcodes[barcodes['z'] == z]
        if len(treeBC) == 0:
            pass
        else:
            tree = cKDTree(treeBC.loc[:, ['x', 'y']].values)
            for compZ in zToCompare:
                queryBC = barcodes[barcodes['z'] == compZ]
                if len(queryBC) == 0:
                    pass
                else:
                    dist, idx = tree.query(queryBC.loc[:, ['x', 'y']].values,
                                           k=1, distance_upper_bound=maxDist)
                    currentHits = treeBC.index.values[idx[np.isfinite(dist)]]
                    comparisonHits = queryBC.index.values[np.isfinite(dist)]
                    graph.add_edges_from(list(zip(currentHits, comparisonHits)))
        connectedComponents = [list(x) for x in
                               list(nx.connected_components(graph))]

    def choose_brighter_barcode(barcodes, indexes):
        sortedBC = barcodes.loc[indexes, :].sort_values(by='mean_intensity',
                                                        ascending=False)
        return sortedBC.index.values.tolist()[0]

    keptBarcodes = barcodes.loc[sorted([x[0] if len(x) == 1 else
                                        choose_brighter_barcode(barcodes, x)
                                        for x in connectedComponents]), :] # type: ignore
    return keptBarcodes

def extract_molecules_from_decoded_image(decoded_image:np.ndarray, pixel_mag:np.ndarray, pixel_traces:np.ndarray, 
                                        distances:np.ndarray, fov:int, crop_width:int, z_index:int=None, 
                                        minimum_area:int=0, barcode_idx:Optional[np.ndarray]=None) -> pd.DataFrame:
    # extract barcodes from a decoded image
    if barcode_idx is None:
        barcode_idx = np.unique(decoded_image)
        barcode_idx = barcode_idx[barcode_idx != -1]
    return pd.concat([extract_molecules_with_index(i, decoded_image, pixel_mag, pixel_traces, distances, 
                                               fov, crop_width, z_index, 
                                               minimum_area) 
                                            for i in barcode_idx])

def extract_molecules_with_index(
        barcodeIndex: int, decodedImage: da.Array,
        pixelMagnitudes: np.ndarray, pixelTraces: np.ndarray,
        distances: np.ndarray, fov: int, cropWidth: int, zIndex: int = None,
        minimumArea: int = 0
) -> pd.DataFrame:
    """Extract the barcode information from the decoded image for barcodes
    that were decoded to the specified barcode index.

    Args:
        barcodeIndex: the index of the barcode to extract the corresponding
            barcodes
        decodedImage: the image indicating the barcode index assigned to
            each pixel
        pixelMagnitudes: an image containing norm of the intensities for
            each pixel across all bits after scaling by the scale factors
        pixelTraces: an image stack containing the normalized pixel
            intensity traces
        distances: an image indicating the distance between the normalized
            pixel trace and the assigned barcode for each pixel
        fov: the index of the field of view
        cropWidth: the number of pixels around the edge of each image within
            which barcodes are excluded from the output list.
        zIndex: the index of the z position
        globalAligner: the aligner used for converted to local x,y
            coordinates to global x,y coordinates
        minimumArea: the minimum area of barcodes to identify. Barcodes
            less than the specified minimum area are ignored.
    Returns:
        a pandas dataframe containing all the barcodes decoded with the
            specified barcode index
    """
    properties = measure.regionprops(
        measure.label(decodedImage == barcodeIndex),
        intensity_image=pixelMagnitudes,
        cache=False)
    is3D = len(pixelTraces.shape) == 4

    columnNames = ['barcode_id', 'fov', 'mean_intensity', 'max_intensity',
                    'area', 'mean_distance', 'min_distance', 'x', 'y', 'z', 'cell_index']
                    #'global_x', 'global_y', 'global_z', 'cell_index']
    if is3D:
        intensityColumns = ['intensity_{}'.format(i) for i in
                            range(pixelTraces.shape[1])]
    else:
        intensityColumns = ['intensity_{}'.format(i) for i in
                            range(pixelTraces.shape[0])]
    if len(properties) == 0:
        return pd.DataFrame(columns=columnNames + intensityColumns)

    allCoords = [list(p.coords) for p in properties]

    if is3D:
        centroidCoords = np.array(
            [prop.weighted_centroid for prop in properties])
        centroids = centroidCoords[:, [0, 2, 1]]
        d = [[distances[y[0], y[1], y[2]] for y in x] for x in allCoords]
        intensityAndAreas = np.array([[x.mean_intensity,
                                        x.max_intensity,
                                        x.area] for x in properties])
        intensities = [
            [pixelTraces[y[0], :, y[1], y[2]] for y in x] for x in
            allCoords]
        intensities = pd.DataFrame(
            [np.mean(x, 0) if len(x) > 1 else x[0] for x in intensities],
            columns=intensityColumns)

    else:
        intensityAndCoords = [
            np.array([[y[0], y[1], pixelMagnitudes[y[0], y[1]]] for y in x])
            for x in allCoords]
        centroidCoords = np.array(
            [[(r[:, 0] * (r[:, -1] / r[:, -1].sum())).sum(),
                (r[:, 1] * (r[:, -1] / r[:, -1].sum())).sum()]
                if r.shape[0] > 1 else [r[0][0], r[0][1]]
                for r in intensityAndCoords])
        centroids = np.zeros((centroidCoords.shape[0], 3))
        centroids[:, 0] = zIndex
        centroids[:, [1, 2]] = centroidCoords[:, [1, 0]]
        d = [[distances[y[0], y[1]] for y in x] for x in allCoords]
        intensityAndAreas = np.array([[x[:, 2].mean(),
                                        x[:, 2].max(),
                                        x.shape[0]]
                                        for x in intensityAndCoords])
        intensities = [[pixelTraces[:, y[0], y[1]] for y in x] for
                        x in allCoords]
        intensities = pd.DataFrame(
            [np.mean(x, 0) if len(x) > 1 else x[0] for x in intensities],
            columns=intensityColumns)

    #globalCentroids = centroids

    df = pd.DataFrame(np.zeros((len(properties), len(columnNames))),
                            columns=columnNames)
    df['barcode_id'] = barcodeIndex
    df['fov'] = fov
    df.loc[:, ['mean_intensity', 'max_intensity', 'area']] = \
        intensityAndAreas
    df.loc[:, ['mean_distance', 'min_distance']] = np.array(
        [[np.mean(x), np.min(x)] if len(x) > 1 else [x[0], x[0]] for x in
            d])
    df.loc[:, ['x', 'y', 'z']] = centroids[:, [1, 2, 0]]
    #df.loc[:, ['global_x', 'global_y', 'global_z']] = \
    #    globalCentroids[:, [1, 2, 0]]
    df['cell_index'] = -1
    fullDF = pd.concat([df, intensities], axis=1)
    fullDF = fullDF[(fullDF['x'].between(cropWidth,
                                            decodedImage.shape[0] - cropWidth,
                                            inclusive='neither')) &
                    (fullDF['y'].between(cropWidth,
                                            decodedImage.shape[1] - cropWidth,
                                            inclusive='neither')) &
                    (fullDF['area'] >= minimumArea)]

    return fullDF

def extract_molecule_counts(molecules:pd.DataFrame, intensity_bins:np.ndarray, distance_bins:np.ndarray, area_bins:np.ndarray) -> np.ndarray:
    molecule_data = molecules[["mean_intensity", "min_distance", "area"]].values
    molecule_data[:,0] = np.log10(molecule_data[:,0])
    return np.histogramdd(molecule_data, bins=(intensity_bins, distance_bins, area_bins))[0]

def extract_molecule_counts_spots(molecules:pd.DataFrame, intensity_bins:np.ndarray, distance_bins:np.ndarray) -> np.ndarray:
    """
    Extract molecule counts for spots without area 
    """
    molecule_data = molecules[["mean_intensity", "min_distance"]].values
    molecule_data[:,0] = np.log10(molecule_data[:,0])
    return np.histogramdd(molecule_data, bins=(intensity_bins, distance_bins))[0]

def molecule_count_histograms(molecules:pd.DataFrame, codebook:Codebook, dist_thresh:float=0.65) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coding and blank barcode histograms.
    Histograms are X x Y x Z tensor of intensity x distance x area
    Barcodes is a dataframe with columns 'barcode_id', 'area', 'distance', 'magnitude'
    """
    coding = codebook.get_coding_indexes()
    blank = codebook.get_blank_indexes()

    area_bins = np.arange(1, 35)
    distance_bins = np.arange(0, dist_thresh+0.02, 0.01)
    fovs = molecules['fov'].unique()
    # randomly sample from at most 20 fields of view
    max_intensity = np.log10(molecules['mean_intensity'].max())
    intensity_bins = np.arange(0, 2*max_intensity, max_intensity/100)
    n_intensity_bins = len(intensity_bins)  
    n_dist_bins = len(distance_bins)
    n_area_bins = len(area_bins)

    blank_counts = np.zeros((n_intensity_bins-1, n_dist_bins-1, n_area_bins-1))
    coding_counts = np.zeros_like(blank_counts)
    for i in fovs:
        curr_mols = molecules[molecules['fov']==i]
        blank_counts += extract_molecule_counts(curr_mols[curr_mols['barcode_id'].isin(blank)], intensity_bins, distance_bins, area_bins)
        coding_counts += extract_molecule_counts(curr_mols[curr_mols['barcode_id'].isin(coding)], intensity_bins, distance_bins, area_bins)
    return coding_counts, blank_counts, intensity_bins, distance_bins, area_bins

def molecule_count_histograms_spots(molecules:pd.DataFrame, codebook:Codebook, dist_thresh:float=0.65) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute coding and blank barcode histograms.
    Histograms are X x Y x Z tensor of intensity x distance x area
    Barcodes is a dataframe with columns 'barcode_id', 'area', 'distance', 'magnitude'
    """
    coding = codebook.get_coding_indexes()
    blank = codebook.get_blank_indexes()

    distance_bins = np.arange(0, dist_thresh+0.02, 0.01)
    fovs = molecules['fov'].unique()
    # randomly sample from at most 20 fields of view
    max_intensity = np.log10(molecules['mean_intensity'].max())
    intensity_bins = np.arange(0, 2*max_intensity, max_intensity/100)
    n_intensity_bins = len(intensity_bins)  
    n_dist_bins = len(distance_bins)

    blank_counts = np.zeros((n_intensity_bins-1, n_dist_bins-1))
    coding_counts = np.zeros_like(blank_counts)
    for i in fovs:
        curr_mols = molecules[molecules['fov']==i]
        blank_counts += extract_molecule_counts_spots(curr_mols[curr_mols['barcode_id'].isin(blank)], intensity_bins, distance_bins)
        coding_counts += extract_molecule_counts_spots(curr_mols[curr_mols['barcode_id'].isin(coding)], intensity_bins, distance_bins)
    return coding_counts, blank_counts, intensity_bins, distance_bins


def blank_fraction_histogram(coding_histogram:np.ndarray, blank_histogram:np.ndarray, n_coding_indexes:int, n_blank_indexes:int) -> np.ndarray:
    """ Get the normalized blank fraction histogram indicating the
    normalized blank fraction for each intensity, distance, and area
    bin.

    Returns: The normalized blank fraction histogram. The histogram
        has three dimensions: mean intensity, minimum distance, and area.
        The bins in each dimension are defined by the bins returned by
        get_area_bins, get_distance_bins, and get_area_bins, respectively.
        Each entry indicates the number of blank barcodes divided by the
        number of coding barcodes within the corresponding bin
        normalized by the fraction of blank barcodes in the codebook.
        With this normalization, when all (both blank and coding) barcodes
        are selected with equal probability, the blank fraction is
        expected to be 1.
    """
    # following the merlin code this is the coding histogram
    total_histogram = coding_histogram# + blank_histogram
    blank_fraction = blank_histogram / total_histogram 
    blank_fraction[total_histogram == 0] = np.finfo(blank_fraction.dtype).max
    blank_fraction /= n_blank_indexes/(n_blank_indexes + n_coding_indexes)
    return blank_fraction

def calculate_misidentification_rate_for_threshold(threshold: float, coding_histogram:np.ndarray, 
                                                   blank_histogram:np.ndarray, blank_fraction:np.ndarray, 
                                                   n_coding_indexes:int, n_blank_indexes:int ) -> float:
    """ Calculate the misidentification rate for a specified blank
    fraction threshold.

    Args:
        threshold: the normalized blank fraction threshold
    Returns: The estimated misidentification rate, estimated as the
        number of blank barcodes per blank barcode divided
        by the number of coding barcodes per coding barcode.
    """
    select_bins = blank_fraction < threshold
    coding_counts = np.sum(coding_histogram[select_bins])
    blank_counts = np.sum(blank_histogram[select_bins])

    return ((blank_counts/n_blank_indexes) /
            (coding_counts/n_coding_indexes))

def calculate_threshold_for_misidentification_rate(target_misidentification_rate: float, 
                                                   coding_histogram:np.ndarray,
                                                   blank_histogram:np.ndarray,
                                                   blank_fraction:np.ndarray,
                                                   n_coding_indexes:int,
                                                   n_blank_indexes:int,
                                                   tolerance:float=0.001) -> float:
    """ Calculate the normalized blank fraction threshold that achieves
    a specified misidentification rate.

    Args:
        targetMisidentificationRate: the target misidentification rate
    Returns: the normalized blank fraction threshold that achieves
        targetMisidentificationRate
    """
    def misidentification_rate_error_for_threshold(x):
        return calculate_misidentification_rate_for_threshold(x, coding_histogram, blank_histogram, blank_fraction, n_coding_indexes, n_blank_indexes) \
            - target_misidentification_rate
    return optimize.newton(
        misidentification_rate_error_for_threshold, 0.2, tol=tolerance, x1=0.3,
        disp=False)

def calculate_barcode_count_for_threshold(threshold: float, coding_histogram:np.ndarray, blank_histogram:np.ndarray, blank_fraction:np.ndarray) -> float:
    """ Calculate the number of barcodes remaining after applying
    the specified normalized blank fraction threshold.

    Args:
        threshold: the normalized blank fraction threshold
    Returns: The number of barcodes passing the threshold.
    """
    return np.sum(blank_histogram[blank_fraction < threshold]) \
        + np.sum(coding_histogram[blank_fraction < threshold])
