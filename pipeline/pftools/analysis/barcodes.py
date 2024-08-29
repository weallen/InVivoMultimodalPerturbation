from tqdm import tqdm
from scipy.stats import zscore
import anndata as ad
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from scipy.sparse import csc_matrix

def compute_mols_per_cell(adata:ad.AnnData) -> pd.DataFrame:
    """
    Create a dataframe of all molecules assigned to all cells
    """
    expr = adata.X.copy()
    gene_names = list(adata.var_names)
    cell_names = list(adata.obs.index)
    cells = []
    mols = []
    for cell_idx in range(expr.shape[0]):
        for bc_id in range(expr.shape[1]):
            if expr[cell_idx, bc_id] > 0:
                for _ in range(int(expr[cell_idx, bc_id])):
                    cells.append(cell_idx)
                    mols.append(bc_id)
    return np.array(cells), np.array(mols), cell_names, gene_names

def count_mols_per_cell(cells:np.ndarray, mols:np.ndarray) -> np.ndarray:
    """
    Count the number of molecules per cell
    """
    uniq_cells = np.unique(cells)
    uniq_mols = np.unique(mols)
    counts = np.zeros((len(uniq_cells), len(uniq_mols)))
    for i, c in enumerate(uniq_cells):
        curr_mols = mols[cells==c]
        curr_uniq = np.unique(curr_mols)
        for j, m in enumerate(curr_uniq):
            counts[i,m] = np.sum(curr_mols == m)
    return csc_matrix(counts)

def count_shuffled_mols_per_cell(cells:np.ndarray, mols:np.ndarray) -> np.ndarray:
    curr_mols = mols.copy()
    np.random.shuffle(curr_mols)
    return count_mols_per_cell(cells, curr_mols)

def zscore_counts(counts:np.ndarray) -> np.ndarray:
    """
    Z-score the counts for each cell
    """
    return zscore(counts, axis=1)

def normalize_counts(counts:np.ndarray) -> np.ndarray:
    """
    Normalize the counts for each cell
    """
    return counts/np.sum(counts, axis=1)[:,None]

def bootstrap_barcode_pvals(adata:ad.AnnData, n_shuffle:int=100,random_seed:int=42) -> ad.AnnData:
    """
    Returned the bootstrapped p-values for each barcode for each cell 
    """
    np.random.seed(random_seed)
    cells, mols, cell_names, mol_names = compute_mols_per_cell(adata)
    counts = count_mols_per_cell(cells, mols).toarray()
    #counts = normalize_counts(counts)
    #counts = zscore(counts,axis=0)
    shuf_counts_all = Parallel(n_jobs=-1)(delayed(count_shuffled_mols_per_cell)(cells, mols) for _ in range(n_shuffle))
    #for i in tqdm(range(n_shuffle)):
    #    np.random.shuffle(mols)
    #    shuf_counts = count_mols_per_cell(cells, mols)
    #    if i == 0:
    #        shuf_counts_all = shuf_counts
    #    else:
    #        shuf_counts_all = np.dstack([shuf_counts_all, shuf_counts])
    #pvals = np.ones_like(counts)
    #for i in range(counts.shape[0]):
    #    for j in range(counts.shape[1]):
    #        pvals[i,j] = np.mean(shuf_counts_all[i,j,:] > counts[i,j])

    pval = np.zeros_like(counts)
    for i in range(len(shuf_counts_all)):
        curr_counts = shuf_counts_all[i].toarray()
        #curr_counts = zscore(curr_counts, axis=0)
        #curr_counts = normalize_counts(curr_counts)
        pval += (curr_counts > counts).astype(int)
    pval /= len(shuf_counts_all)
    pval_corrected = pval.copy()
    for i in range(pval_corrected.shape[0]):
        pval_corrected[i,:] = multipletests(pval[i,:], method='fdr_bh')[1]
    #return counts, shuf_counts_all
    return counts, pval, pval_corrected
    #return pvals, pvals_corrected.reshape(pvals.shape)

def call_barcodes():
    pass