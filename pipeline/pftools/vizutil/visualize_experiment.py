import argparse
import os
import sys

from pftools.vizutil.napari_viz import *

def parse_args(args):
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize a set of tiles.')
    parser.add_argument('base_path', type=str, help='Path to the base directory.')
    parser.add_argument('base_name', type=str, help="Base name of the tiles.") 
    parser.add_argument('n_colors', type=int, help="Number of colors in the tiles.")
    parser.add_argument('max_projection', type=bool, default=False, help="Whether to max project the tiles.")
    parser.add_argument('contrast_lims', type=float, nargs=2, default=[0, 65535], help='Contrast limits.')
    parser.add_argument('tile_id', type=int, help='Tile')
    parsed_args = parser.parse_args(args)
    return parsed_args

def main(args):
    viewer = napari.Viewer()
    for fpath in args.fpaths:
        tile = xr.open_dataarray(fpath)
        viewer.add_image(tile, name=os.path.basename(fpath))
    napari.run()

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))