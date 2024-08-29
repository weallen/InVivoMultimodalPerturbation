#!/usr/bin/python
import sys
import argparse
import os
from pftools.pipeline.core.experiment import BaseExperiment
from pftools.pipeline.util.cmdutil import parse_basic_args, handle_cluster_at_launch
from pftools.pipeline.util.utils import create_logger
from pftools.pipeline.processing.molecules import FastMoleculeAssigner
import geopandas as geo
import pandas as pd

def main():
    arg_parser = argparse.ArgumentParser()
    # add specific args for molecules and cells
    arg_parser.add_argument("--molecule_path", type=str, default=None, help="Path to molecule data.")
    arg_parser.add_argument("--output_prefix", type=str, default=None, help="Prefix of output data.")
    arg_parser.add_argument("--cell_path", type=str, default=None, help="Path to cell segmentation.")
    arg_parser.add_argument("--data", type=str, help="Input data directory")
    arg_parser.add_argument("--output", type=str, help="Output directory")
    arg_parser.add_argument('--dataorg', type=str, default=None, help="Data organization")
    arg_parser.add_argument('--microscope', type=str, default=None, help="Microscope type")
    arg_parser.add_argument('--codebook', type=str, default=None, help="Codebook path")
    arg_parser.add_argument('--positions', type=str, default=None, help="Positions path")
    arg_parser.add_argument('--output_name', type=str, default=None, help="Output name")
    arg_parser.add_argument("--cluster_address", default=None, type=str, required=False, help="Address of the cluster")
    arg_parser.add_argument("--cluster_port", default=None, type=str, required=False, help="Port of the cluster")
    arg_parser.add_argument("--n_workers", default=None, type=int, required=False, help="Number of workers")
    arg_parser.add_argument("--threads_per_worker", default=2, type=int, required=False, help="Number of threads per worker")
    arg_parser.add_argument("--memory_limit", default=None, type=str, required=False, help="Memory limit per worker")
    arg_parser.add_argument('--tasks_per_worker', default=1.0, type=float, required=False, help="Multiplier for number of workers to use")
    #args = parse_basic_args(sys.argv[1:], arg_parser)
    args = arg_parser.parse_args()

    # set up logger
    logger = create_logger(args.output, 'assign_molecules')

    # load the experiment
    logger.info("Loading experiment...")
    experiment = BaseExperiment(args.data, args.output, args.dataorg, args.microscope, args.codebook)
    
    # set up client
    client = handle_cluster_at_launch(args)
    print(client)

    # load the molecules
    logger.info("Loading molecules...")
    if args.molecule_path is not None:
        if "parquet" in args.molecule_path:
            molecules = pd.read_parquet(args.molecule_path)
        elif "csv" in args.molecule_path:
            molecules = pd.read_csv(args.molecule_path)
        else:
            sys.exit("ERROR: Molecule path must be parquet or csv")
    else:
        sys.exit("ERROR: No molecule path provided")
    
    # load the cells
    logger.info("Loading cells...")
    if args.cell_path is not None:
        cells = geo.read_parquet(args.cell_path)
    else:
        sys.exit("ERROR: No cell path provided")

    # run the assignment
    logger.info("Running assignment...")
    if args.output_name is not None:
        output_path = os.path.join(experiment.output_path, args.output_name)
    else:
        # use default name
        output_path = os.path.join(experiment.output_path, "assigned_molecules")

    molecule_assigner = FastMoleculeAssigner(
        client=client,
        codebook=experiment.codebook,
        molecules=molecules,
        cells=cells,
        global_align=experiment.global_align,
        output_path=output_path,
        output_prefix=args.output_prefix
    )
    molecule_assigner.run()

if __name__ == "__main__":
    main()
