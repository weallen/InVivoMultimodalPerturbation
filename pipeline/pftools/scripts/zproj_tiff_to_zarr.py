# Convert an experiment from normal Hal format of TIFF files to Zarr, Z-projecting each stack
import os
import numpy as np
import tifffile
import zarr
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
from numcodecs import Blosc

# example filename: primary-f42-r2-c3-z0.tiff
def get_info(filenames, field):
    return np.max([int(f.split(".")[0].split('-')[field][1:]) for f in filenames])+1

def main():
    output_file = "test.zarr"
    
    # save single zarr file for each tile (FOV)
    # structure of zarr is 
    # /primary/r_R/raw/ch_CH
    # /primary/r_R/max/ch_CH
    # and optionally /primary/round_R/nuclei and /R/nuclei_max
    # as well as /round_R/fiducial and /R/fiducial_max
    #
    # convert from default starfish format to zarr

    parser = argparse.ArgumentParser()
    #parser.add_argument("-r", "--nrounds", type=int, help="Number of rounds to process")
    #parser.add_argument("-c" , "--nchannel", type=int, help="Number of channels")
    #parser.add_argument("-z", "--nz", type=int, help="Number of Z planes")
    parser.add_argument("-i","--input", type=str, help="Input directory")
    parser.add_argument("-o", "--output", type=str, help="Output directory")

    args = parser.parse_args()

    print(f"Loading data from {args.input} and writing to {args.output}")

    files = [i for i in os.listdir(args.input) if "tiff" in i]
    n_fov = get_info(files, 1)
    n_r = get_info(files, 2)
    n_c = get_info(files, 3)
    n_z = 1 #get_info(files, 4)

    is_max_proj = (n_z == 1)
    if is_max_proj:
        print("Found only max projection...")
    print(f"Found FOV: {n_fov}, Rounds: {n_r}, Colors: {n_c}, Z: {n_z}...")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # get n_x and n_y from test file
    print("Converting data...")
    for i in tqdm(range(n_fov)):
        #  load data for current FOV

        root = zarr.open(os.path.join(args.output, f"tile_{i}.zarr"), 'w')
#        primary = root.create_group("primary")
        # only works for max projected now
        #curr_data_max = np.zeros((n_r, n_c, 2048, 2048), dtype="uint16")
        primary = root.create_group("primary")
        nuclei = root.create_group("nuclei")
        protein = root.create_group("protein")
        max_dset = primary.create_dataset("max", shape=(n_r, n_c, 2048, 2048), chunks=(1,1,2048,2048), dtype="uint16", compressor=compressor)
        max_nuclei = nuclei.create_dataset("max", shape=(n_r, 2048, 2048), chunks=(1,2048,2048), dtype="uint16", compressor=compressor)
        max_protein = protein.create_dataset("max", shape=(n_c, 2048, 2048), chunks=(1,2048,2048), dtype="uint16", compressor=compressor)
        for r in range(n_r):
            for c in range(n_c):
                max_dset[r,c,:,:] = tifffile.imread(os.path.join(args.input, f"primary-f{i}-r{r}-c{c}-z0.tiff"))

        for r in range(n_r):
            max_nuclei[r,:,:] = tifffile.imread(os.path.join(args.input, f"nuclei-f{i}-r{r}-c0-z0.tiff"))

        for c in range(n_c):
            max_protein[c,:,:] = tifffile.imread(os.path.join(args.input, f"protein-f{i}-c{c}-z0.tiff")) 

if __name__ == "__main__":
    main() 



    