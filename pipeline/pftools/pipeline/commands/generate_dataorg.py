#!/usr/bin/env python
import argparse
import sys
import os
from pftools.pipeline.util.data import generate_dataorg_for_threecolor_experiment

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--output", type=str, help="Path to data organization file")    
    args.add_argument("--nz", type=int, default=1, help="Number of z planes")
    args.add_argument("--nbits", type=int, default=16, help="Number of bits")
    args.add_argument("--ncols", type=int, default=5, help="Number of colors")
    args.add_argument("--add_dapi", action=argparse.BooleanOptionalAction, default=True, help="Add DAPI channel")
    args.add_argument("--correct_750_zoffset", action=argparse.BooleanOptionalAction, default=False, help="Correct for 750 z-offset")
    args.add_argument("--image_type", type=str, default="zscan_5cols_slow", help="Image type")
    args.add_argument("--image_reg_exp", type=str, default="(?P<imageType>[\w|-]+)_(?P<fov>[0-9]+)_(?P<imagingRound>[0-9]+)", help="Image regular expression")
    args.add_argument("--zspace", type=float, default=1., help="Z spacing")
    args.add_argument('--swap_dapi', action='store_true', default=False, help="Change order to have DAPI as first rather than last round")
    args = args.parse_args()
    dorg = generate_dataorg_for_threecolor_experiment(args.nbits, args.nz, args.ncols, args.image_type, 
                                                      args.image_reg_exp, args.correct_750_zoffset, args.add_dapi, z_spacing=args.zspace, swap_dapi=args.swap_dapi) 
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dorg.to_csv(args.output, index=True)

if __name__ == "__main__":
    main()