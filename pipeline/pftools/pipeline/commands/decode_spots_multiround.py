#!/usr/bin/env python
import argparse
import sys
import numpy as np
from pftools.pipeline.core.experiment import BaseExperiment
from pftools.pipeline.util.cmdutil import parse_basic_args, handle_cluster_at_launch, add_args_from_params, get_params_from_args
from pftools.pipeline.util.utils import create_logger
from pftools.pipeline.processing.spot_decoding import MultiRoundSpotDecoder, SpotDecoderParams
import os

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_args_from_params(arg_parser, SpotDecoderParams())

    args = parse_basic_args(sys.argv[1:], arg_parser)

    # get params back
    params = get_params_from_args(args, SpotDecoderParams())

    # get logger
    logger = create_logger(args.output, 'decode_spots')

    # load the experiment
    logger.info("Loading experiment...")
    experiment = BaseExperiment(args.data, args.output, args.dataorg, args.microscope, args.codebook, max_fov=args.nfov)

    # launch cluster
    client = handle_cluster_at_launch(args, logger)

    # load the data
    logger.info("Lazily loading data...")
    readout_names = experiment.codebook.get_bit_names()
    if args.nfov is None:
        experiment.load_data(readout_names=readout_names)
    else:
        experiment.load_data(n_fov=args.nfov, readout_names=readout_names)

    # run the decoding
    logger.info("Running decoding...")
    if args.output_name is not None:
        output_path = os.path.join(experiment.output_path, args.output_name)
    else:
        # use default name
        output_path = os.path.join(experiment.output_path, "decoded_spots")

    readout_names = experiment.codebook.get_bit_names()
    # get tile subset corresponding to used bits
    #tiles, fid_tiles = experiment.get_tile_subset_for_readouts(readout_names)

    # get reference for alignment
    #ref_imgs = experiment.get_reference_images_by_readout(args.base_readout)
    #print(tiles[0].shape, fid_tiles[0].shape, ref_imgs[0].shape)
    sd = MultiRoundSpotDecoder(client=client, tiles=experiment.tiles, 
                               output_path=output_path, codebook=experiment.codebook, params=params, 
                               global_align=experiment.global_align)
    sd.run(worker_multiplier=args.tasks_per_worker)

if __name__ == "__main__":
    main()