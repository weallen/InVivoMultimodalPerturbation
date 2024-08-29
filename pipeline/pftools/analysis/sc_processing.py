import scanpy as sc
import anndata as ad
import numpy as np
from typing import List
from pftools.analysis.allcools.seurat_class import SeuratIntegration
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from tqdm import tqdm
from scipy.stats import ranksums, ttest_ind, wilcoxon
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
import pandas as pd

def calculate_de_genes_per_group(adata:ad.AnnData, key='cluster_type', groupby:str='cond', reference:str='adlib', use_raw:bool=True, test:str='ranksums', n_reps:int=10, n_subsample:int=250) -> None:
    de_genes_df = []
    if use_raw:
        gene_names = list(adata.raw.var_names)
    else:
        gene_names = list(adata.var_names)
    for i in adata.obs[key].unique():
        print(i)
        for n in range(n_reps):
            curr_adata = adata[adata.obs[key] == i]
            curr_adata = curr_adata[np.random.choice(curr_adata.shape[0], n_subsample),:]
            pvals, fc = calculate_de_genes(curr_adata, groupby=groupby, reference=reference, use_raw=use_raw, test=test)
            de_genes_df.append(pd.DataFrame({'clust_type':[i]*len(pvals), 'pval':pvals, 'fc': fc, 'gene':gene_names, 'rep':[n]*len(pvals)}))
    return pd.concat(de_genes_df)

def calculate_de_genes(adata:ad.AnnData, groupby:str='cond', reference:str='adlib', use_raw:bool=True, test:str='ranksums') -> None:
    if use_raw:
        temp = adata.raw.to_adata()
        temp.X = temp.X.todense()
    else:
        temp = adata.copy()
    a1 = temp[temp.obs[groupby] != reference]
    a2 = temp[temp.obs[groupby] == reference]
    if test == 'ranksums':
        #pvals = Parallel(n_jobs=-1)(delayed(wilcoxon)(a1.X[:,k], a2.X[:,k])[1] for k in range(a1.X.shape[1]))
        pvals = multipletests(np.array([ranksums(a1.X[:,k], a2.X[:,k])[1] for k in tqdm(range(a1.X.shape[1]))]), method='fdr_bh')[1]
    elif test == 'ttest':
        #pvals = Parallel(n_jobs=-1)(delayed(ttest_ind)(a1.X[:,k], a2.X[:,k])[1] for k in range(a1.X.shape[1]))
        pvals = multipletests(np.array([ttest_ind(a1.X[:,k], a2.X[:,k])[1] for k in tqdm(range(a1.X.shape[1]))]), method='fdr_bh')[1]
    fold_change = a1.X.mean(axis=0)/a2.X.mean(axis=0)
    return pvals, fold_change

def preprocess_data(adata:ad.AnnData, min_counts:int=100, max_counts:int=1000000, vars_to_regress:list=[]) -> ad.AnnData:
    """
    Do basic preprocessing of data to get ready for dimensionality reduction. 
    """
    adata.var_names_make_unique()
    # calculate QC metrics 
    sc.pp.calculate_qc_metrics(adata,  percent_top=None, log1p=False, inplace=True)

    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, max_counts=max_counts)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if len(vars_to_regress) > 0:
        sc.pp.regress_out(adata, vars_to_regress)
    sc.pp.scale(adata, max_value=10)
    return adata

def relabel_clusters(adata:ad.AnnData, orig_key:str='new_leiden', new_key:str='new_leiden_relabeled') -> ad.AnnData:
    lookup = {}
    for n,i in enumerate(np.unique(adata.obs[orig_key])):
        lookup[i] = str(n)
    adata.obs[new_key] = [lookup[i] for i in adata.obs[orig_key]]
    return adata

def merge_small_clusters(adata:ad.AnnData, key:str='leiden') -> ad.AnnData:
    pass

def greedy_merge_clusters(adata:ad.AnnData, key:str='leiden', n_de_thresh:int=10) -> ad.AnnData:
    clust_labels = np.array(adata.obs[key].values)
    clust_lookup = {}
    for i in sorted(np.unique(clust_labels)):
        for j in sorted(np.unique(clust_labels)):
            a1 = adata[adata.obs[key]==i]
            a2 = adata[adata.obs[key]==j]
            if a2.shape[0] > 0 and a1.shape[0] > 0:
                pvals = multipletests(np.array([ranksums(a1.X[:,k], a2.X[:,k])[1] for k in range(a1.X.shape[1])]), method='fdr_bh')[1]
                n_de = np.sum((a1.X.mean(axis=0)/a2.X.mean(axis=0) > 2) & (pvals < 0.01))
                print(i,j, n_de)
                if n_de < 10:
                    clust_labels[clust_labels == j] = i
                    clust_lookup[j] = i
    new_labels = {}
    for i, l in enumerate(np.unique(clust_labels)):
        new_labels[l] = i
    clust_labels = np.array([new_labels[l] for l in clust_labels])

    #new_lookup = {}
    #for k, v in clust_lookup.items():
    #    new_lookup[k] = new_labels[v]
    return clust_labels, clust_lookup 

def impute_data(adata_joint:ad.AnnData, adata_raw:ad.AnnData, obs_col:str='datatype', key1:str='merfish', 
                key2:str='flex', n_neighbors:int=10, n_pcs:int=50,
                expr_thresh:float=0.0) -> ad.AnnData:
    """
    Impute the data from adata_raw into adata_joint, using nearest neighbors for each cell in key1 to key2 
    """
    # for each cell in adata_joint, find the n_neighbors nearest neighbors in adata2
    output = adata_joint.copy()
    adata1 = adata_joint[adata_joint.obs[obs_col] == key1]
    adata2 = adata_joint[adata_joint.obs[obs_col] == key2]
    kdtree = KDTree(adata2.obsm['X_pca'][:, :n_pcs])
    # split into chunks for NN search
    indices = []
    n_chunk = 10000
    n_cells = adata1.shape[0]
    for i in tqdm(range(0, n_cells, n_chunk)):
        dists, idx = kdtree.query(adata1.obsm['X_pca'][i:i+n_chunk], k=n_neighbors)
        indices.extend(idx)
    # for each cell in adata1, find the n_neighbors nearest neighbors in adata2
    imputed_data = np.zeros((n_cells, adata_raw.shape[1]))
    for i in tqdm(range(n_cells)):
        imputed_data[i] = np.mean(adata_raw.X[indices[i],:], axis=0)
    imputed_data[imputed_data < expr_thresh] = 0
    output = ad.AnnData(X=imputed_data, obs=adata1.obs[:n_cells], var=adata_raw.var)
    return output

def reduce_dimensionality(adata:ad.AnnData, n_comps:int=100) -> ad.AnnData:
    sc.pp.pca(adata, svd_solver='arpack', n_comps=n_comps)
    return adata

def compute_umap(adata, n_pcs=50, n_neighbors=10) -> ad.AnnData:
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata)
    return adata

def cluster_cells(adata: ad.AnnData, resolution:float=1.0) -> ad.AnnData:
    sc.tl.leiden(adata, resolution=resolution)
    return adata

def harmony_integrate(adata: ad.AnnData, batch_key:str='batch', swap_pca:bool=True):
    sc.external.pp.harmony_integrate(adata, batch_key=batch_key)
    if swap_pca:
        adata.obsm['X_pca_orig'] = adata.obsm['X_pca']
        adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
    return adata

def allcools_integrate(adata_merged:ad.AnnData, adata_list:List[ad.AnnData], n_pcs:int=40, cell_type_col:str=None) -> ad.AnnData:
    integrator = SeuratIntegration()
    integrator.find_anchor(adata_list,
                        k_local=None,
                        key_local='X_pca',
                        k_anchor=5,
                        key_anchor='X',
                        dim_red='cca',
                        max_cc_cells=100000,
                        k_score=30,
                        k_filter=None, #why?
                        scale1=False,
                        scale2=False,
                        n_components=n_pcs,
                        n_features=200,
                        alignments=[[[0], [1]]])
    # coembed the data
    corrected = integrator.integrate(key_correct='X_pca',
                                 row_normalize=True,
                                 n_components=n_pcs,
                                 k_weight=100,
                                 sd=1,
                                 alignments=[[[0], [1]]])

    adata_merged.obsm['X_pca_integrate'] = np.concatenate(corrected)

    # transfer labels if necessary
    if cell_type_col is not None:
        print("Performing label transfer")
        transfer_results = integrator.label_transfer(
            ref=[0],
            qry=[1],
            categorical_key=[cell_type_col],
            key_dist='X_pca',
            kweight=100,
            npc=n_pcs
        )
        integrator.save_transfer_results_to_adata(adata_merged, transfer_results)

    return adata_merged

