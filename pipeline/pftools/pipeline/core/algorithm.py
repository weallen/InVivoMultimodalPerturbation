from abc import ABCMeta, abstractmethod
from typing import Optional, List, Tuple
import logging
import dask.array as da
import xarray as xr
import dask.distributed as dd
import sys
import os
import dask.distributed as dd
import time
from threading import Thread
import numpy as np
from functools import partial
import psutil

from pftools.pipeline.util.utils import create_logger
from pftools.pipeline.core.tile import TileLoader

def get_total_memory_gb():
    return psutil.virtual_memory().total/1e9

class AnalysisTask(object):
    def __init__(self, output_path:str):
        super().__init__()
        #self.output_base_path = output_path
        self.output_path = output_path #os.path.join(self.output_base_path, self.__class__.__name__)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.logger = self._create_logger()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def _create_logger(self) -> logging.Logger:
        """
        Create a logger for the decoding.
        """
        return create_logger(self.output_path, self.__class__.__name__)

    def __enter__(self):
        # Initialize resources or setup
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            # Log the exception
            self.logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
            # Handle the exception, if necessary
            return True  # Returning True prevents the exception from propagating

class ParallelAnalysisTask(AnalysisTask):
    """
    Class to apply some operation in parallel using Dask cluster.
    """
    def __init__(self, client: dd.Client, output_path:str):
        super().__init__(output_path)
        self.client = client

class ParallelTileAnalysisTask(AnalysisTask):
    """
    Base class to apply some algorithm to a list of tiles. 
    This using a Dask cluster to submit jobs, then runs threads in the background to handle the results.
    """
    def __init__(self, client:dd.Client, tiles:List[TileLoader], output_path:str):
        super().__init__(output_path)
        self.client = client
        self.tiles = tiles

    def run(self, worker_multiplier:int=-1, poll_delay:int=1, memory_per_worker_gb:int=15) -> None:
        """
        Run the algorithm on the tiles, splitting it into chunks.
        This keeps a fixed number of chunks running in parallel, to avoid overloading the cluster and to keep the task graph small.
        The results of the computation are handled by a thread that saves out the results as they are finished.
        """
        # run whatever happens before run
        self._prerun_processing()

        # run loop
        start_time = time.time()
        self.logger.info("Starting run...")
        n_workers = len(self.client.scheduler_info()['workers'].keys())
        self.logger.info(f"Number of workers: {n_workers}")
        n_tiles = len(self.tiles)
        if worker_multiplier == -1:
            memory = get_total_memory_gb()
            n_tasks = min(int(memory) // int(memory_per_worker_gb), n_tiles)

            self.logger.info(f"Automatically setting number of workers based on {int(memory)} GB available RAM")
            self.logger.info(f"Using {memory_per_worker_gb} to start {n_tasks} tasks")
        else:
            n_tasks = min(int(np.round(worker_multiplier*n_workers)), n_tiles)
        all_tile_idx = list(np.arange(n_tiles))

        self.logger.info(f"Processing {n_tiles} tiles, processing {n_tasks} tasks concurrently")
        # launch initial tasks
        # save out results of optimization
        futures = []
        n_completed = 0
        # launch first n_tasks 
        for i in range(n_tasks):
            curr_tile_idx = all_tile_idx.pop(0)
            self.logger.info(f"Starting processing of tile {curr_tile_idx}")
            futures.append(self._process_tile(curr_tile_idx))
            time.sleep(poll_delay)
            # stagger the initial jobs

        # this runs a loop that monitors the progress of the decoding
        # and saves out the results as they are finished 
        while len(futures) > 0:
            # create a thread to monitor the progress of the decoding
            # this is necessary because the decoding can take a long time
            # and we want to be able to save out the results as they are finished
            for i,(tile_idx, f) in enumerate(futures):
                # only add a task when one finishes
                ok_to_add = False
                if f is not None:
                    if f.done():
                        n_completed += 1
                        self.logger.info(f"Finished processing tile {tile_idx} in %0.02f minutes. {n_completed} done, {n_tiles-n_completed} tiles remaining" % float((time.time() - start_time)/60))
                        # get the results back in the main loop
                        self._tile_processing_done_callback(tile_idx, f)
                        #t = Thread(target=self._tile_processing_done_callback, args=(tile_idx, f,)) 
                        #saving_threads.append(t.start())
                        # block until this is complete
                        #t.join()
                        # remove from the list of futures
                        temp = futures.pop(i)
                        del temp
                        ok_to_add = True

                    elif f.cancelled():
                        self.logger.warning(f"Tile {tile_idx} was cancelled and did not complete processing.")
                        futures.pop(i)
                        n_completed += 1
                        ok_to_add = True
                else:
                    temp = futures.pop(i)
                    del temp
                    n_completed += 1
                    ok_to_add = True
                # launch the next task then break until next loop
                if len(all_tile_idx) > 0 and ok_to_add:
                    curr_tile_idx = all_tile_idx.pop(0)
                    self.logger.info(f"Starting processing of tile {curr_tile_idx}")
                    futures.append(self._process_tile(curr_tile_idx))
                
            # poll every poll_delay seconds
            time.sleep(poll_delay)
        #self.logger.info("Joining output handler threads...")
        #saving_threads = [t.join() for t in saving_threads if t is not None]
        self.logger.info("Finished processing...")

        # run whatever happens afterewards
        self._postrun_processing()
        return

    def _prerun_processing(self) -> None:
        """
        Do any processing before the main run loop.
        """
        pass

    def _postrun_processing(self) -> None:
        """
        Do any processing after the main run loop.
        """
        pass

    @abstractmethod 
    def _tile_processing_done_callback(self, tile_idx:int, future:dd.Future) -> None:
        """
        Save the results of a computation for a tile. 
        Use future.result() to get the results of the computation.
        """
        raise NotImplementedError()

    @abstractmethod 
    def _process_tile(self, tile_idx:int) -> Tuple[int, dd.Future]:
        """
        Process a single tile. This is called by client.submit, which returns a future containing the results of the computation. 
        """
        raise NotImplementedError()

