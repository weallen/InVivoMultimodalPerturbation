import argparse
import sys
import os
from pftools.pipeline.core.experiment import BaseExperiment
from pftools.pipeline.util.cmdutil import parse_basic_args, handle_cluster_at_launch, add_args_from_params, get_params_from_args
from pftools.pipeline.util.utils import create_logger
from pftools.pipeline.processing.segment import CellPoseSegmenter, CellSegmentationParameters, ExtractAndCleanFeatures
from pftools.pipeline.util.cellpose_segment import handle_mask_flips    

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_args_from_params(arg_parser, CellSegmentationParameters())

    args = parse_basic_args(sys.argv[1:], arg_parser)

    # get params back
    params = get_params_from_args(args, CellSegmentationParameters())

    # get logger
    logger = create_logger(args.output, 'segment_cells')

    # load the experiment
    logger.info("Loading experiment...")
    experiment = BaseExperiment(args.data, args.output, args.dataorg, args.microscope, args.codebook, max_fov=args.nfov)

    # zeroth readout is reference for registration
    base_channel = experiment.data_organization.get_data_info_df().readout_name.iloc[0]
    # launch cluster
    client = handle_cluster_at_launch(args, logger)

    # load the data
    logger.info("Lazily loading data...")
    if args.cyto_chan is None or args.nuclei_chan is None:
        raise ValueError("Must provide cyto and nuclei channels")
    # assume first readout is reference and include that
    elif args.cyto_chan is not None and args.nuclei_chan is not None:
        readout_names = [args.cyto_chan, args.nuclei_chan]
    elif args.cyto_chan is not None and args.nuclei_chan is None:
        readout_names = [args.cyto_chan]
    elif args.cyto_chan is None and args.nuclei_chan is not None:
        readout_names = [args.nuclei_chan]
    if base_channel not in readout_names:
        readout_names = [base_channel] + readout_names

    logger.info(f"Segmenting using {readout_names[1:]}. Registering using {base_channel}")
    if args.nfov is None:
        experiment.load_data(readout_names=readout_names)
    else:
        experiment.load_data(n_fov=args.nfov, readout_names=readout_names)

    # run the segmentation
    logger.info("Running segmentation...")
    if args.output_name is not None:
        output_path = os.path.join(experiment.output_path, args.output_name)
    else:
        # use default name
        output_path = os.path.join(experiment.output_path, "segment_cells")

    # run segmentation on registered tiles
    seg = CellPoseSegmenter(client, experiment.tiles, output_path, params)
    seg.run(worker_multiplier=args.tasks_per_worker, poll_delay=1)

    # load the masks
    logger.info("Loading masks...")
    masks = seg.load_masks()

    #logger.info("Handling mask flips...")
    #masks = handle_mask_flips(masks, experiment)
 
    # get masks
    #masks = da.compute(masks)[0]
    # clean up the segmentation
    logger.info("Running segmentation...")
    cleaner = ExtractAndCleanFeatures(client, experiment.output_path, masks, experiment.global_align)
    cleaner.run()


if __name__ == "__main__":
    main()
