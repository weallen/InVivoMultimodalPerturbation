import argparse
import sys
import os
from pftools.pipeline.core.experiment import BaseExperiment
from pftools.pipeline.util.cmdutil import parse_basic_args, handle_cluster_at_launch, add_args_from_params, get_params_from_args
from pftools.pipeline.util.utils import create_logger
from pftools.pipeline.processing.merfish import RCAPixelBasedDecoder, PixelBasedDecodeParams

def main():
    arg_parser = argparse.ArgumentParser()
    # add a few extra params for optimization
    arg_parser.add_argument("--optimize_load_factors", action=argparse.BooleanOptionalAction, help="Load pre-computed scale factors from output directory.")
    arg_parser.add_argument("--optimize_n_fov", type=int, default=50, help="Number of FOV for optimization")
    arg_parser.add_argument("--optimize_n_iter", type=int, default=1, help="Number of iterations for optimization. (Default is no optimization)")

    # add arguments for each parameter to decoder
    arg_parser = add_args_from_params(arg_parser, PixelBasedDecodeParams())

    args = parse_basic_args(sys.argv[1:], arg_parser)

    # get params back
    params = get_params_from_args(args, PixelBasedDecodeParams())

    # get logger
    logger = create_logger(args.output, 'decode_pixels')

    # load the experiment
    logger.info("Loading experiment...")
    experiment = BaseExperiment(args.data, args.output, args.dataorg, args.microscope, args.codebook, max_fov=args.nfov)

    # launch cluster
    client = handle_cluster_at_launch(args, logger)

    # load the data

    readout_names = experiment.codebook.get_bit_names()
    logger.info("Lazily loading data...")
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
        output_path = os.path.join(experiment.output_path, "decoded_pixels")

    #tiles, fid_tiles = experiment.get_tile_subset_for_readouts(readout_names)
    #ref_imgs = experiment.get_reference_images_by_readout(args.base_readout)

    pixel_decoder = RCAPixelBasedDecoder(client=client, 
                                        tiles=experiment.tiles, 
                                        output_base_path=output_path, 
                                        params=params,
                                        codebook=experiment.codebook, 
                                        global_align=experiment.global_align)
    if args.optimize_load_factors:
        logger.info("Loading scale factors...")
        if not pixel_decoder.load_scale_factors():
            pixel_decoder.optimize(n_iter=args.optimize_n_iter, n_fov=args.optimize_n_fov)
    else:
        pixel_decoder.optimize(n_iter=args.optimize_n_iter, n_fov=args.optimize_n_fov)
    
    # run decoding
    pixel_decoder.run(worker_multiplier=args.tasks_per_worker)

if __name__ == "__main__":
    main()
