
from typing import List,Optional
from glob import glob
import pandas as pd
import zarr
import os
from tifffile import imread
from tqdm import tqdm
from numcodecs import Blosc, LZMA
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import warnings

def identify_position_file(tiff_dir:str) -> Optional[str]:
    """
    Identify the position file in the tiff directory.
    """
    pos_file = None
    for f in os.listdir(tiff_dir):
        if "tiled_pos" in f or "pos_tiled" in f:
            pos_file = os.path.join(tiff_dir, f)
            break
    return pos_file

def make_tiff_file_structure_df(tiff_filenames: List[str]) -> List[str]:
    """
    Given the list of tiff filenames with the format prefix_fov_round.tif, make a dataframe of all the tiff filenames, prefixes, fovs, and rounds.
    """
    prefixes = []
    fovs = []
    rounds = []

    for tiff_filename in tiff_filenames:
        fname = tiff_filename.split(".")[0]
        parts = fname.split("_")
        curr_fov = parts[-2]
        curr_round = parts[-1]
        prefix = "_".join(parts[:-2])
        prefixes.append(prefix)
        fovs.append(int(curr_fov))
        rounds.append(int(curr_round))
    return pd.DataFrame({'filename':tiff_filenames,
                         'prefix':prefixes, 
                         'fovs':fovs, 
                         'rounds':rounds})

def validate_tiff_prefixes_and_range(tiff_prefixes_and_range: pd.DataFrame) -> bool:
    """
    Validate the tiff prefixes and range to ensure that the range of fov and round is consistent for each prefix.
    """
    
    for prefix in tiff_prefixes_and_range['prefix'].unique():
        curr_prefix_df = tiff_prefixes_and_range[tiff_prefixes_and_range['prefix'] == prefix]
        min_fov = curr_prefix_df['fovs'].min()
        max_fov = curr_prefix_df['fovs'].max()
        min_round = curr_prefix_df['rounds'].min()
        max_round = curr_prefix_df['rounds'].max()
        n_fov = len(curr_prefix_df['fovs'].unique())
        n_round = len(curr_prefix_df['rounds'].unique())
        if n_fov != (max_fov - min_fov + 1):
            raise ValueError(f"Range of fov {(min_fov, max_fov)} is not consistent with {n_fov} for prefix {prefix}")
        if n_round != (max_round - min_round + 1):
            raise ValueError(f"Range of round {(min_round, max_round)} is not consistent with {n_round} for prefix {prefix}")
        print(f"Prefix {prefix.split('/')[-1]} has {n_fov} fovs and {n_round} rounds")
    return True
        
def determine_tiff_structure(tiff_dir:str, tiff_pattern:str='*.tif') -> List[str]:
    """
    Determine the tiffs in the tiff_dir that match the tiff_pattern.
    """
    tiffs = glob(os.path.join(tiff_dir, tiff_pattern))
    tiff_df = make_tiff_file_structure_df(tiffs)
    if validate_tiff_prefixes_and_range(tiff_df):
        return tiff_df
    else:
        raise ValueError("Invalid tiff structure")
    
#def check_file_size(tiff_df:pd.DataFrame):
#    file_sizes = {}
#    for t in tiff_df['prefix'].unique():
#        prefix_df = tiff_df[tiff_df['prefix'] == t]
#        curr_fname = prefix_df['filename'].iloc[0]
#        file_sizes[prefix_df] = imread(curr_fname).shape
#    return file_sizes

def identify_all_nontiff_files(tiff_dir:str) -> List[str]:
    """
    Identify all non-tiff files in the tiff directory.
    """
    all_files = os.listdir(tiff_dir)
    tiff_files = glob(os.path.join(tiff_dir, '*.tif'))
    non_tiff_files = [f for f in all_files if f not in tiff_files]
    return [i for i in non_tiff_files if not i.startswith('.') and not i.startswith('..') and not os.path.isdir(i)]

def save_database(root:zarr.Group, tiff_org:pd.DataFrame) -> zarr.Group:
    # Convert the list of dictionaries to a DataFrame
    db = tiff_org.loc[:, ['prefix','fov','round']].sort_values(by=['prefix', 'fov', 'round'])
    root.attrs.update({'db': db.to_dict()})
    return root

def _write_to_zarr_helper(curr_round_group:zarr.Array, group_name:str, fname:str, compressor:Blosc=None, use_compression:bool=False) -> None:
    #print("Writing", group_name, "to zarr...")
    try:
        data = imread(fname).astype(np.uint16)
        #curr_round_group = group.create_group(group_name)
        if use_compression:
            curr_dset = curr_round_group.create_dataset(group_name, 
                                    data=data, 
                                    shape=data.shape,
                                    chunks=(1,data.shape[1],data.shape[2]),
                                    compressor=compressor,
                                    dtype=data.dtype)
        else:
            curr_dset = curr_round_group.create_dataset(group_name, 
                                    data=data, 
                                    shape=data.shape,
                                    chunks=(1,data.shape[1],data.shape[2]),
                                    dtype=data.dtype)
        base_name = fname.split(".")[0]
        power_fname = base_name + ".power"
        xml_fname = base_name + ".xml"
        off_fname = base_name + ".off"
        if os.path.exists(power_fname):
            curr_dset.attrs['power'] = open(power_fname, 'r').read()
        if os.path.exists(xml_fname):
            curr_dset.attrs['xml'] = open(xml_fname, 'r').read()
        if os.path.exists(off_fname):
            curr_dset.attrs['off'] = open(off_fname, 'r').read()
        return
    except Exception as e:
        print("Error writing", group_name, "to zarr:", e)

def convert_tiffs_to_zarr(tiff_dirname:str, zarr_path:str, pos_file:Optional[str]=None, use_compression:bool=True) -> None:
    print("Converting tiffs to zarr...")
    print("Determining tiff structure...")    
    tiff_df = determine_tiff_structure(tiff_dirname)

    print("Identifying non-tiff files...")
    non_tiff_files = identify_all_nontiff_files(tiff_dirname)
#    file_sizes = check_file_size(tiff_df)
    if use_compression:
        compressor = Blosc(cname='lz4', clevel=9, shuffle=Blosc.BITSHUFFLE)
    if zarr_path.endswith('.zarr'):
        store = None
        root = zarr.open(zarr_path, mode='w')
    elif zarr_path.endswith('.zip'):
        store = zarr.ZipStore(zarr_path, mode='w')
        root = zarr.group(store=store)

    #pool = multiprocessing.Pool(multiprocessing.cpu_count())
    if pos_file is not None:
        print("Loading position file:", pos_file)
        pos = pd.read_csv(pos_file, sep=',', header=None, names=['x', 'y'])
        root.create_dataset('pos', data=pos.values)
    for prefix in tiff_df['prefix'].unique():
        print("Processing prefix:", prefix.split('/')[-1])
        print("Found FOVs:", len(tiff_df[tiff_df['prefix'] == prefix]['fovs'].unique()))
        curr_group = root.create_group(prefix.split('/')[-1])
        prefix_df = tiff_df[tiff_df['prefix'] == prefix]
        print("Launching jobs...")
        for fov in tqdm(prefix_df['fovs'].unique()):
            curr_fov_group = curr_group.create_group(f'fov_{fov}')
            fov_df = prefix_df[prefix_df['fovs'] == fov]
            filenames = []
            rounds = []
            #jobs = []
            for r in fov_df['rounds'].unique():
                round_df = fov_df[fov_df['rounds'] == r]
                curr_fname = round_df['filename'].iloc[0]
                #jobs.append(pool.apply_async(_write_to_zarr_helper, (curr_fov_group, f'round_{r}', curr_fname, compressor, use_compression)))
                # remove all the extra files from non_tiff_files (power, etc)
                non_tiff_files = [f for f in non_tiff_files if curr_fname.split(".")[0] not in f] 
                filenames.append(curr_fname)
                rounds.append(r)
                #_write_to_zarr_helper(curr_fov_group, f'round_{r}', curr_fname, compressor, use_compression)
            if zarr_path.endswith('.zarr'):    
                Parallel(n_jobs=multiprocessing.cpu_count() // 2)(delayed(_write_to_zarr_helper)(curr_fov_group, f'round_{rounds[i]}', filenames[i], compressor, use_compression) for i in range(len(filenames)))
            else:
                # can't have parallel write for zip store
                for i in range(len(filenames)):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _write_to_zarr_helper(curr_fov_group, f'round_{rounds[i]}', filenames[i], compressor, use_compression)
        #print("Processing jobs...")
        #for j in tqdm(jobs):
        #    j.get()
    #pool.close()
    print("Generating file organization database...")
    save_database(root, tiff_df)
    # save out the remaining non tiff files
    print("Saving out remaining non-tiff files...")
    for f in non_tiff_files:
        print(f)
        if f.endswith('.xml') or f.endswith('.off') or f.endswith('.power'):
            pass
        else:
            if not os.path.isdir(f):
                root.attrs[f] = open(os.path.join(tiff_dirname, f), 'r').read()
    #zarr.consolidate_metadata(zarr_path)

    # close zip store if it exists
    if store is not None:
        store.close()