import numpy as np
import argparse

def process_fov(fov_path:str, codebook:np.ndarray) -> np.ndarray:
    """
    Decode a single field of view
    """
    # Load the field of view
    fov = np.load(fov_path)
    # Decode the field of view
    decoded = decode_fov(fov, codebook)
    return decoded

def main():
    args = argparse.ArgumentParser(description='Decode barcodes from a set of images.')
    args.add_argument('--input', type=str, required=True)
    args.add_argument('--codebook', type=str, required=True)
    args.add_argument('--output', type=str, required=True)
    args = args.parse_args()

    # Create the input file map

    # Load the codebook

    # Register images

    # Decode the barcodes

if __name__ == "__main__":
    main()