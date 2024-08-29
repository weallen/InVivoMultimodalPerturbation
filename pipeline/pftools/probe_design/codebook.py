from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.spatial.distance import hamming
import urllib 
from MERFISH_probe_design.IO.file_io import write_merlin_codebook

# Step 2: Compute the covering tables for the transcriptome and rRNA/tRNAs
#
def generate_initial_codebook_assignment(gene_names:List[str], codewords:np.ndarray, random_seed:int=40) -> pd.DataFrame:
    """
    Generate an initial codebook assignment with blanks, given a list of gene names.
    """

    blanks = [f'Blank-{n}' for n in range(codewords.shape[0]-len(gene_names))]
    gene_names_with_blanks = gene_names + blanks
    # set random seed
    np.random.seed(random_seed)
    codewords = codewords[np.random.permutation(codewords.shape[0]),:]
    
    return pd.DataFrame({'name': gene_names_with_blanks, 'barcode_str' : stringify_binary_mat_codebook(codewords[:len(gene_names_with_blanks),:])})

def generate_sequential_code(n_bits:int, n_codewords:int) -> np.ndarray:
    """
    Generate a sequential codebook
    """
    codebook = np.zeros((n_codewords, n_bits))
    for i in range(n_codewords):
        codebook[i,i] = 1
    return codebook

def convert_code_list(code_list:List[int], n_bits:int) -> np.ndarray:
    """
    Convert a list of codes to a binary codebook 
    """
    n_codeword = len(code_list)
    print("N codewords:", n_codeword, "N bits:", n_bits)
    codebook = np.zeros((n_codeword, n_bits))
    for i,c in enumerate(code_list):
        for j in c:
            codebook[i,j] = 1
    return codebook

def save_codebook(codewords:np.ndarray, gene_names:List[str], bit_names:List[str], fpath:str, codebook_name:Optional[str]=None) -> None:
    if codebook_name is None:
        codebook_name = "merfish_codebook"
    # Generate the codebook assignment file
    n_codes, n_code_bits = codewords.shape
    n_genes = len(gene_names)
    print(f"Generating codebook for {n_genes} genes and {n_code_bits} bits. {n_codes} codes in the codebook.")
    #print(gene_list['gene_symbol'].tolist())
    codebook = generate_initial_codebook_assignment(gene_names, codewords)

    # Save the codebook assignment file as a MERlin codebook
    codebook_file = fpath
    codebook_version = '1.0'
    # get the names of the bits that are actually used based on 
    used_bit_idx = np.argwhere(np.any(codewords, axis=0)).ravel()
    bit_names_used = [bit_names[i] for i in used_bit_idx]

    transcript_ids = codebook['name'].tolist()
    # Save the codebook
    #print(codebook['name'].tolist())
    write_merlin_codebook(codebook_file, codebook_version, codebook_name, bit_names_used, 
                          codebook['name'].tolist(), transcript_ids, 
                          codebook.barcode_str.tolist())

# Function to download covering tablea
# Example here: https://ljcr.dmgordon.org/cover/show_cover.php?v=26&k=4&t=3
def download_covering_table(k:int, t:int, v:int, min_hd:int=4) -> None:
    """
    Download a covering table from the web
    """
    url = f"https://ljcr.dmgordon.org/cover/show_cover.php?v={v}&k={k}&t={t}"
    fpath = f"/tmp/covering_table_{k}_{t}_{v}.txt" 
    urllib.request.urlretrieve(url, fpath)
    # blank lines and HTML tags
    with open(fpath, "r") as f:
        lines = f.readlines()
    lines = [l for l in lines if l != "\n" and "<" not in l]
    with open(fpath, "w") as f:
        f.writelines(lines)
    covering_table = load_covering_table(fpath, n_on_bits=k)
    return covering_table#eliminate_hd_cutoff(covering_table, min_hd=min_hd)

def load_covering_table(fpath:str, n_on_bits:int=4) -> np.ndarray:
    temp = pd.read_csv(fpath,header=None, sep="\s+",engine="python")
    temp.columns = [f"bit{i+1}" for i in range(n_on_bits)]
    bin_mat = np.zeros((temp.shape[0], temp[f"bit{n_on_bits}"].max()))
    for i,r in temp.iterrows():
        for k in range(4):
            bin_mat[i, r[f"bit{k+1}"]-1] = 1
    return bin_mat 

def compute_hd_mat(codewords:np.ndarray) -> np.ndarray:
    """
    Compute the hamming distance matrix between codewords 
    """
    n_codewords, n_bits = codewords.shape

    hd_mat = np.zeros((n_codewords, n_codewords))
    for i in range(n_codewords):
        for j in range(n_codewords):
            if i == j:
                hd_mat[i,j] = np.inf
            else:
                hd_mat[i,j] = hamming(codewords[i,:], codewords[j,:])*n_bits
    return hd_mat

def eliminate_hd_cutoff(codebook:np.ndarray, min_hd:int=4) -> np.ndarray:
    """
    Greedily remove codeworks below a HD cutoff
    """
    n_codewords, n_bits = codebook.shape
    hd_mat = compute_hd_mat(codebook)
    all_idx = list(range(codebook.shape[0]))
    # remove all bad indices, then add back
    bad_idx_pairs = np.unique(np.argwhere(hd_mat < min_hd).flatten())
    good_idx = [i for i in range(n_codewords) if i not in bad_idx_pairs]
    # greedily try adding bad bad indices
    for i in bad_idx_pairs:
        curr_codebook = codebook[np.array(good_idx).astype(int),:]
        curr_hd = np.array([hamming(codebook[i,:], curr_codebook[j,:])*n_bits for j in range(curr_codebook.shape[0])])
        if np.all(curr_hd >= min_hd):
            good_idx.append(i)
    final_codebook = codebook[np.array(good_idx).astype(int),:] 
    assert compute_hd_mat(final_codebook).min() >= min_hd
    return final_codebook

def stringify_binary_mat_codebook(m:np.ndarray) -> List[str]:
    """
    Convert a binary matrix to a list of strings of 0s and 1s
    """
    return ["".join([str(int(i)) for i in m[j,:]]) for j in range(m.shape[0])]
