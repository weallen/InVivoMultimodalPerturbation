#!/usr/bin/env python

# Using a cellular segmentation, export a cropped image of protein expression for each cell
# and save to a zarr with annotation of the cell id and the protein name for deepmorph analysis

import argparse
import sys
import os
from tqdm import tqdm
import dask.array as da
from dask_image.imread import imread
import geopandas as geo
from pftools.pipeline.core.experiment import BaseExperiment
from pftools.pipeline.util.cmdutil import parse_basic_args, handle_cluster_at_launch, add_args_from_params, get_params_from_args
from pftools.pipeline.util.utils import create_logger
from pftools.pipeline.export.image_export import ImageExporter, ImageExportParameters

import warnings
warnings.filterwarnings('ignore')

def lazily_load_masks(masks_path:str, n_fov:int):
    """
    Load the masks lazily
    """
    return [imread(os.path.join(masks_path, f'tile_{i}_cellpose_seg.tif')) for i in tqdm(range(n_fov))]

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_args_from_params(arg_parser, ImageExportParameters())
    arg_parser.add_argument("--cell_feats", type=str, required=True, help="Path to the cell feature data")
    arg_parser.add_argument("--masks", type=str, required=True, help="Path to the exported masks data")

    args = parse_basic_args(sys.argv[1:], arg_parser)

    # get params back
    params = get_params_from_args(args, ImageExportParameters())

    # get logger
    logger = create_logger(args.output, 'image_export')

    # load the experiment
    logger.info("Loading experiment...")
    experiment = BaseExperiment(args.data, args.output, args.dataorg, args.microscope, args.codebook, max_fov=args.nfov)

    # launch cluster
    client = handle_cluster_at_launch(args, logger)

    # load the data
    logger.info("Lazily loading data...")
    experiment.load_data()

    # load the cell features
    logger.info("Loading cell features...")
    cell_feats = geo.read_parquet(args.cell_feats)

    # load the masks
    logger.info("Loading masks...")
    masks = lazily_load_masks(args.masks, experiment.positions.shape[0])

    # run the segmentation
    logger.info("Running protein export...")
    if args.output_name is not None:
        output_path = os.path.join(experiment.output_path, args.output_name)
    else:
        # use default name
        output_path = os.path.join(experiment.output_path, "image_export")

    # export proteins
    export = ImageExporter(client, experiment.tiles, masks, cell_feats, experiment.global_align,
                           experiment.microscope_information, experiment.data_organization, output_path, params)
    export.run(worker_multiplier=args.tasks_per_worker)

if __name__ == "__main__":
    main()
