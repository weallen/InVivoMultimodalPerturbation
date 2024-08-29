import argparse
import sys
from typing import List
from pftools.pipeline.core.experiment import BaseExperiment
from pftools.pipeline.util.cmdutil import parse_basic_args, handle_cluster_at_launch, add_args_from_params, get_params_from_args
from pftools.pipeline.util.utils import create_logger
from pftools.pipeline.processing.segment import ExtractAndCleanFeatures
from dask_image.imread import imread
import dask.array as da
from pftools.pipeline.util.cellpose_segment import handle_mask_flips, load_masks

def main():
    arg_parser = argparse.ArgumentParser()
    args = parse_basic_args(sys.argv[1:], arg_parser)

    # get logger
    logger = create_logger(args.output, 'convert_masks_to_segs')

    # load the experiment
    logger.info("Loading experiment...")
    experiment = BaseExperiment(args.data, args.output, args.dataorg, args.microscope, args.codebook, args.positions)

    # launch cluster
    client = handle_cluster_at_launch(args, logger)
    # load the masks
    logger.info("Loading masks...")
    masks = load_masks(args.data, experiment.get_num_fovs())
    masks = handle_mask_flips(masks, experiment)

   
    # get masks
    #masks = da.compute(masks)[0]
    # clean up the segmentation
    logger.info("Running segmentation...")
    cleaner = ExtractAndCleanFeatures(client, experiment.output_path, masks, experiment.global_align)
    cleaner.run()

if __name__ == "__main__":
    main()