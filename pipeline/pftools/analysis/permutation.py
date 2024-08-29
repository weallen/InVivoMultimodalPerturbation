# Adapted from XingJie Pan's code: 
# https://raw.githubusercontent.com/xingjiepan/MERFISH_analysis/189059306d9537bf1bf455a603af69e0f7d944da/cell_cell_contact/permutation.py

import numpy as np
import pandas as pd
import scipy.spatial as spatial
from scipy.sparse import csr_matrix


def find_neighbors_of_points(points, radius):
    point_tree = spatial.cKDTree(points)
    return point_tree.query_ball_point(points, radius)

def generate_global_permutation(N_cells):
    '''Permutate the cells.
    Return a list of permutated cell IDs.
    '''
    permutated_cell_ids = list(range(N_cells))
    np.random.shuffle(permutated_cell_ids)
    return permutated_cell_ids

def generate_local_permutation(neighbors):
    '''Permutate the cells that are neighbors.
    Return a list of permutated cell IDs.
    '''
    permutated_cell_ids = list(range(len(neighbors)))
    remaining_cell_ids = set(range(len(neighbors)))
    
    for i in range(len(neighbors)):
        if i in remaining_cell_ids:
            j = np.random.choice(neighbors[i])
            permutated_cell_ids[i] = j
            permutated_cell_ids[j] = i
            
            remaining_cell_ids.discard(i)
            remaining_cell_ids.discard(j)
    
    return permutated_cell_ids

def count_cell_type_contacts(neighbors, cell_type_ids_of_cells, N_cell_types, permutated_cell_ids=None,
                             use_sparse_matrix=False):
    '''Count the contacts between cell types.
    Return a sparse matrix of the number of contacts between
    pairs of cell types.
    '''
    # If no permutation is specified, use identical permutation
    if None is permutated_cell_ids:
        permutated_cell_ids = list(range(len(neighbors)))
    
    # Initialize the counting matrix
    if use_sparse_matrix:
        ct_contact_count_mtx = csr_matrix((N_cell_types, N_cell_types), dtype=int)
    else:
        ct_contact_count_mtx = np.zeros((N_cell_types, N_cell_types), dtype=int)
    
    # Count the contacts
    for i in range(len(neighbors)):
        p_i = permutated_cell_ids[i]
        ct_i = cell_type_ids_of_cells[p_i]
        
        for j in neighbors[i]:
            if i != j: # Ignore the cell itself
                p_j = permutated_cell_ids[j]
                ct_j = cell_type_ids_of_cells[p_j]
            
                ct_contact_count_mtx[ct_i, ct_j] += 1
            
    return ct_contact_count_mtx

def generate_cell_type_contact_count_matrices(df, cell_type_col, coord_cols, cell_types, 
                                        permutation_method, N_permutations=1, contact_radius=15,
                                        local_permute_radius=50):
    '''Generate a cell type contact matrix.
    Args:
        df: The pandas DataFrame of one slice of tissue.
        cell_type_col: The column in df that defines cell types.
        coord_cols: A list of columns in df that define the spatial coordinates.
        cell_types: A list of cell types.
        permutation_method: The method for permutation allowed values are 
            ['no_permutation', 'local_permutation', 'global_permutaion'].
        N_permutations: The number of permutations to perform.
        contact_radius: The radius within with two cells are regarded as close contacts.
        local_permute_radius: The radius for local permutation.
    '''
    assert(permutation_method in ['no_permutation', 'local_permutation', 'global_permutation'])

    # Find the contacting neighbors of each cell
    points = np.array(df[coord_cols])
    point_tree = spatial.cKDTree(points)
    neighbors_contact = point_tree.query_ball_point(points, contact_radius)

    # Get the cell type id of each cell
    N_cell_types = len(cell_types)
    cell_type_ids = list(range(N_cell_types))
    cell_type_id_map = pd.DataFrame({'cell_type_id':cell_type_ids, 'cell_type':cell_types})
    cell_type_ids_of_cells = list(df[[cell_type_col]].merge(cell_type_id_map, 
            left_on=cell_type_col, right_on='cell_type', how='left')['cell_type_id'])

    if permutation_method == 'no_permutation':
        return count_cell_type_contacts(neighbors_contact, cell_type_ids_of_cells, N_cell_types)
        
    elif permutation_method == 'global_permutation':
        permuted_contact_tensor = np.zeros((N_permutations, N_cell_types, N_cell_types))
        
        for i in range(N_permutations):
            permutated_cell_ids = generate_global_permutation(len(cell_type_ids_of_cells))
            permuted_ct_contact_count_mtx = count_cell_type_contacts(neighbors_contact, cell_type_ids_of_cells,
                                                            N_cell_types, permutated_cell_ids=permutated_cell_ids)
            permuted_contact_tensor[i] = permuted_ct_contact_count_mtx

        return permuted_contact_tensor

    elif permutation_method == 'local_permutation':
        neighbors_permute = point_tree.query_ball_point(points, local_permute_radius)
        permuted_contact_tensor = np.zeros((N_permutations, N_cell_types, N_cell_types))
       
        for i in range(N_permutations):
            permutated_cell_ids = generate_local_permutation(neighbors_permute)
            permuted_ct_contact_count_mtx = count_cell_type_contacts(neighbors_contact, cell_type_ids_of_cells,
                                                            N_cell_types, permutated_cell_ids=permutated_cell_ids)
            permuted_contact_tensor[i] = permuted_ct_contact_count_mtx

        return permuted_contact_tensor