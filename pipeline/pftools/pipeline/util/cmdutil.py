from typing import Optional, List
from dask.distributed import Client, LocalCluster
import multiprocessing
import argparse
import logging
import os
from dataclasses import dataclass 

def add_args_from_params(parser:argparse.ArgumentParser, params:object) -> argparse.ArgumentParser:
    for name, field in params.__dataclass_fields__.items():
        if field.type == bool:
            parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=field.default, help=f"Parameter: {name} ({field.type}), default {field.default}")
        else:
            parser.add_argument(f"--{name}", type=field.type, default=field.default, help=f"Parameter: {name} ({field.type}), default {field.default}")
    return parser

def get_params_from_args(args:argparse.Namespace, params:object) -> object:
    for name in params.__dataclass_fields__.keys():
        setattr(params, name, getattr(args, name))
    return params

def parse_basic_args(args:List[str], parser:Optional[argparse.ArgumentParser]=None) -> argparse.Namespace:
    if parser is None:
        parser = argparse.ArgumentParser()
    
    # add default arguments
    parser.add_argument("--data", type=str, help="Input data directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--dataorg", type=str, default=None, help="Data organization file")
    parser.add_argument("--microscope", type=str, default=None, help="Microscope information file")
    parser.add_argument("--codebook", type=str, default=None, help="Codebook file")
    parser.add_argument("--positions", type=str, default=None, help="Positions file")
    parser.add_argument("--output_name", type=str, default=None, help="Name of the outputs")
    parser.add_argument('--nfov', type=int, default=None, help="Number of fields of view to process")
    parser.add_argument('--base_readout', type=str, default="DAPI", help="Readout name for channel to use for registration. Everything will be aligned to the round with this readout.")

    # stuff for cluster
    parser.add_argument("--cluster_address", default=None, type=str, required=False, help="Address of the cluster")
    parser.add_argument("--cluster_port", default=None, type=str, required=False, help="Port of the cluster")

    parser.add_argument("--n_workers", default=None, type=int, required=False, help="Number of workers")
    parser.add_argument("--threads_per_worker", default=2, type=int, required=False, help="Number of threads per worker")
    parser.add_argument("--memory_limit", default=None, type=str, required=False, help="Memory limit per worker")

    parser.add_argument('--tasks_per_worker', default=1.0, type=float, required=False, help="Multiplier for number of workers to use")
    results = parser.parse_args(args)
    return results

def handle_cluster_at_launch(args, logger:logging.Logger=None) -> Client:
    if args.n_workers is None:
        # use the number of cpu cores
        n_workers = multiprocessing.cpu_count() // 2 - 1 # leave some cores for the OS, and use only physical cores
    else:
        n_workers = args.n_workers
    if args.threads_per_worker is None:
        threads_per_worker = 2
    else:
        threads_per_worker = args.threads_per_worker
    if args.cluster_address is None:
        client = launch_cluster(n_workers, threads_per_worker)
        if logger is not None:
            logger.info(f"Launching cluster with {n_workers} workers, {threads_per_worker} threads per worker...")
            logger.info(f"Dashboard at {client.dashboard_link}")
    else:
        if logger is not None:
            logger.info(f"Connecting to cluster at {args.cluster_address}:{args.cluster_port}...")
        client = connect_to_cluster(args.cluster_address, args.cluster_port)
    return client

def launch_cluster(n_workers:Optional[int]=None, threads_per_worker:int=2) -> Client:

    #if memory_limit is None:
        # use the total memory
    #    memory_limit = '{}GB'.format(int((multiprocessing.virtual_memory().total / 1e9)/n_workers))

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)
    client = Client(cluster)
    return client

def connect_to_cluster(ip_address:str, port:int) -> Client:
    client = Client('{}:{}'.format(ip_address, port))
    return client

