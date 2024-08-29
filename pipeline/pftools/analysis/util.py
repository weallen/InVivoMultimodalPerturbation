from typing import Tuple, Optional
import anndata as ad
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def remove_blanks_from_adata(adata:ad.AnnData) -> ad.AnnData:
    """
    Remove Blank genes from AnnData object.
    """
    adata = adata[:, [i for i in adata.var_names if not i.startswith('Blank')]]
    return adata

def concatenate_adata(adata1: ad.AnnData, adata2: ad.AnnData) -> ad.AnnData:
    """
    Merge two AnnData objects by cells with the same coordinates.
    """
    adata = ad.AnnData(
        X = np.concatenate([adata1.X, adata2.X], axis=0),
        obs = adata1.obs.append(adata2.obs),
        var = adata1.var
    )
    return adata

def merge_adata_by_coords(adata1: ad.AnnData, adata2: ad.AnnData, min_dist:float=0.1) -> ad.AnnData:
    """
    Merge two AnnData objects by cells with the same coordinates, based on mutual nearest neighbor.
    """
    # find the mutual nearest neighbor for each cell in adata2
    ind1, ind2 = mutual_nearest_neighbors_for_adata(adata1, adata2, min_dist)
    adata1_nn = adata1[ind1,:]
    adata2_nn = adata2[ind2,:]
    adata = ad.AnnData(
        X = np.concatenate([adata1_nn.X, adata2_nn.X], axis=1),
        obs = adata1_nn.obs,
    )
    adata.var_names = adata1.var_names.tolist() + adata2.var_names.tolist()
    return adata

def nearest_neighbors_for_adata(adata1: ad.AnnData, adata2: ad.AnnData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the mutual nearest neighbor for each cell in adata2.
    """
    tree = KDTree(adata1.obs[['x', 'y']])
    dist, ind = tree.query(adata2.obs[['x', 'y']])
    return dist, ind

def mutual_nearest_neighbors_for_adata(adata1:ad.AnnData, adata2:ad.AnnData, min_dist:float=0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the mutual nearest neighbor for each cell in adata2.
    """
    dist1, ind1 = nearest_neighbors_for_adata(adata1, adata2)
    dist2, ind2 = nearest_neighbors_for_adata(adata2, adata1)
    mutual1 = np.array([i for i in range(len(ind1)) if ind2[ind1[i]] == i and dist1[i] < min_dist])
    mutual2 = np.array([ind1[i] for i in mutual1])
    return mutual2, mutual1

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

def hierarchical_clustering(data_matrix, method='single', criterion='distance', t=2):
    """
    Perform hierarchical clustering on a data matrix and return cluster labels.

    Parameters:
    - data_matrix (numpy.ndarray): The data matrix to be clustered.
    - method (str): The linkage method to use. Default is 'single'.
    - criterion (str): The criterion to use in forming flat clusters. Default is 'distance'.
    - t (float or int): The threshold to apply when forming flat clusters based on the criterion. Default is 2.

    Returns:
    - numpy.ndarray: An array of cluster labels.
    """
    # Compute hierarchical clustering
    Z = linkage(data_matrix, method=method)

    # Extract cluster labels
    labels = fcluster(Z, t=t, criterion=criterion)

    return labels
