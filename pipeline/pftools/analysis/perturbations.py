import anndata as ad
import numpy as np
from typing import Tuple
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
def mean_expr_by_bc(adata:ad.AnnData, min_cells:int=10, col_key:str='bc', shuffle_bc:bool=False) -> np.ndarray:
    """
    Calculate mean expression for each barcode.
    """
    expr = adata.X.copy()
    bc = adata.obs[col_key].values.copy()
    if shuffle_bc:
        np.random.shuffle(bc)
    valid_bc = np.array([i for i in np.unique(bc) if np.sum(bc==i) > min_cells and i != -1])
    mean_expr = np.vstack([np.mean(expr[bc==i,:], axis=0) for i in valid_bc])
    return mean_expr, valid_bc

def bootstrap_perturb_corrs(adata:ad.AnnData, n_shuffle:int=100, min_cells:int=10, col_key:str='bc') -> Tuple[np.ndarray, np.ndarray]:
    # get the correlation matrix for the mean
    mean_expr, valid_bc = mean_expr_by_bc(adata, min_cells=min_cells, col_key=col_key, shuffle_bc=False)
    corrs = np.corrcoef(mean_expr)
    
    # get the correlation matrix for the shuffled data
    for i in tqdm(range(n_shuffle)):
        mean_expr_shuf, _ = mean_expr_by_bc(adata, min_cells=min_cells, col_key=col_key, shuffle_bc=True)
        corrs_shuf = np.corrcoef(mean_expr_shuf)
        if i == 0:
            corrs_shuf_all = corrs_shuf
        else:
            corrs_shuf_all = np.dstack([corrs_shuf_all, corrs_shuf])

    # calculate the p-value for each correlation
    pvals = np.ones_like(corrs)
    for i in range(corrs.shape[0]):
        for j in range(corrs.shape[1]):
            if i != j:
                pvals[i,j] = np.mean(corrs_shuf_all[i,j,:] > corrs[i,j])
    pvals = multipletests(pvals.flatten(), method='fdr_bh')[1]
    return corrs, pvals.reshape(corrs.shape), valid_bc

def normalize_by_controls(mean_expr:np.ndarray, controls:np.ndarray, zscore:bool=True) -> np.ndarray:
    """
    Normalize the mean expression by the controls.
    """
    if zscore:
        return (mean_expr - mean_expr[controls,:].mean(0))/np.std(mean_expr[controls,:], axis=0)
    else:
        return (mean_expr - mean_expr[controls,:].mean(0))#/np.std(mean_expr[controls,:], axis=0)

def bootstrap_perturb_corrs_pca(adata:ad.AnnData, n_pcs:int=10, n_shuffle:int=100, min_cells:int=10, col_key:str='bc', controls:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
    # get the correlation matrix for the mean
    mean_expr, valid_bc = mean_expr_by_bc(adata, min_cells=min_cells, col_key=col_key, shuffle_bc=False)
    if controls is not None:
        mean_expr = zscore_by_controls(mean_expr, controls)
    mean_expr_pc = PCA(n_components=n_pcs).fit_transform(mean_expr)

    corrs = np.corrcoef(mean_expr_pc)
    
    # get the correlation matrix for the shuffled data
    for i in tqdm(range(n_shuffle)):
        mean_expr_shuf, _ = mean_expr_by_bc(adata, min_cells=min_cells, col_key=col_key, shuffle_bc=True)
        if controls is not None:
            mean_expr_shuf = zscore_by_controls(mean_expr_shuf, controls)
        mean_expr_pc_shuf = PCA(n_components=n_pcs).fit_transform(mean_expr_shuf)
        corrs_shuf = np.corrcoef(mean_expr_pc_shuf)
        if i == 0:
            corrs_shuf_all = corrs_shuf
        else:
            corrs_shuf_all = np.dstack([corrs_shuf_all, corrs_shuf])

    # calculate the p-value for each correlation
    pvals = np.ones_like(corrs)
    for i in range(corrs.shape[0]):
        for j in range(corrs.shape[1]):
            if i != j:
                pvals[i,j] = np.mean(np.abs(corrs_shuf_all[i,j,:]) > np.abs(corrs[i,j]))
    #pvals = multipletests(pvals.flatten(), method='fdr_bh')[1]
    return corrs, pvals.reshape(corrs.shape), valid_bc