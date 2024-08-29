import dask.distributed as dd
import dask.array as da
from dask.delayed import delayed
from distributed import Future
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import zarr
import numpy as np
from skimage import measure
import rtree
import os
from shapely import geometry
from typing import List, Dict
from scipy.spatial import cKDTree
from shapely.geometry import box
from dask.delayed import delayed
import networkx as nx
import pandas as pd
import xarray as xr
import geopandas as geo
from tqdm import tqdm
import tifffile
import time
from pftools.pipeline.core.algorithm import ParallelTileAnalysisTask, ParallelAnalysisTask
from pftools.pipeline.processing.globalalign import SimpleGlobalAlignment
from pftools.pipeline.util.spatialfeature import SpatialFeature, simple_clean_cells, construct_graph, construct_tree, remove_overlapping_cells
from pftools.pipeline.core.tile import TileLoader
from pftools.pipeline.util.cellpose_segment import cellpose_segment, combine_2d_segmentation_masks_into_3d, preprocess_stack_for_segmentation
from pftools.pipeline.processing.registration import register_tile_to_reference
from dask_image.imread import imread

@dataclass
class CellSegmentationParameters():
    use_gpu:bool = False # don't want this for now
    flow_threshold:float = 0.4
    diameter:float = 160.0
    min_size:float = 200.
    channels:int = 1 # 1 for nuclei, 2 for cyto, 3 for both
    nuclei_chan:str = 'DAPI' # this is used for selecting out the channels from each tile, but not in the class itself
    cyto_chan:str = 'polyT'
    model_type:str = 'cyto3'
    max_project:bool = False
    pretrained_model_path:str = None

class CellPoseSegmenter(ParallelTileAnalysisTask):
    def __init__(self, client:dd.Client, tiles:List[TileLoader],
                 output_path:str,
                parameters:CellSegmentationParameters):

        super().__init__(client, tiles, output_path)
        self.params = parameters
        if self.params.channels == 1:
            self.params.channels = (self.params.nuclei_chan, None)
        elif self.params.channels == 2:
            self.params.channels = (None, self.params.cyto_chan)
        elif self.params.channels == 3:
            self.params.channels = (self.params.nuclei_chan, self.params.cyto_chan)
        else:
            raise ValueError("Invalid number of channels specified. Must be 1, 2, or 3.")
        self.n_tiles = len(tiles)
        if os.path.exists(self.params.pretrained_model_path):
            self.logger.info(f"Using model from path: {self.params.pretrained_model_path}")
        else:
            raise ValueError(f"Can't find model at: {self.params.pretrained_model_path}")
    def _process_tile(self, tile_idx:int) -> Tuple[int, dd.Future]:
        #self.logger.info(f"Registering tile {tile_idx} to reference...")
        # register the tile to the reference
        #if self.cyto_chan_tiles[tile_idx] is not None:
        #    cyto_chan_tile, _ = self.cyto_chan_tiles[tile_idx].get_registered_data().data #register_tile_to_reference(self.cyto_chan_tiles[tile_idx].data, self.ref_imgs[tile_idx].data)
        #if self.nuclei_chan_tiles[tile_idx] is not None:
        #    nuclei_chan_tile, _ = self.nuclei_chan_tiles[tile_idx].get_registered_data().data #register_tile_to_reference(self.nuclei_chan_tiles[tile_idx].data, self.ref_imgs[tile_idx].data)
        # rearrange data for segmentation
        # there is probably a more elegant way to deal with the various conditions
        #if cyto_chan_tile is not None and nuclei_chan_tile is None:
        #    channel, to_segment = self._rearrange_data_for_segmentation(cyto_chan_tile, None)
        #elif cyto_chan_tile is None and nuclei_chan_tile is not None:
        #    channel, to_segment = self._rearrange_data_for_segmentation(None, nuclei_chan_tile)
        #elif nuclei_chan_tile is not None and nuclei_chan_tile is not None:
        #    channel, to_segment = self._rearrange_data_for_segmentation(cyto_chan_tile, nuclei_chan_tile)

        # get actual stack and scatter to workers
        #to_segment = to_segment.compute()
        #tifffile.imsave(file=os.path.join(self.output_path, f"tile_{tile_idx}_preseg.tif"), data=to_segment.astype(np.uint16))
        #to_segment = self.client.scatter(to_segment)
        # submit the task
        if not os.path.exists(os.path.join(self.output_path, f"tile_{tile_idx}_cellpose_seg.tif")):
            return tile_idx, self.client.submit(segment_cells,
                                                self.tiles[tile_idx],
                                                self.params.cyto_chan,
                                                self.params.nuclei_chan,
                                                self.params.use_gpu,
                                                self.params.model_type,
                                                self.params.diameter,
                                                self.params.min_size,
                                                self.params.flow_threshold,
                                                self.params.max_project,
                                                self.params.pretrained_model_path)
        else:
            self.logger.info(f"Tile {tile_idx} already segmented. Skipping.")
            return tile_idx, None

    def _tile_processing_done_callback(self, tile_idx:int, future:dd.Future) -> None:
        self.logger.info(f"Tile {tile_idx} done")
        # save out the results
        self._save_results(tile_idx, future.result()) # type: ignore

    def _save_results(self, tile_idx, segmented_masks:np.ndarray) -> None:
        self.logger.info(f"Saving results for tile {tile_idx}")
        # save the labeled mask
        tifffile.imsave(file=os.path.join(self.output_path, f"tile_{tile_idx}_cellpose_seg.tif"), data=segmented_masks.astype(np.uint16))
        # save the spatial features


    def load_masks(self) -> List[da.Array]:
        """
        Lazily load the masks after segmentation.
        """
        masks = [imread(os.path.join(self.output_path, f"tile_{tile_idx}_cellpose_seg.tif")) for tile_idx in tqdm(range(self.n_tiles))]
        return masks

def rearrange_data_for_segmentation(cyto_chan_tile:da.Array, nuclei_chan_tile:da.Array, max_project:bool=False) -> Tuple[Tuple[int, int], da.Array]:
    # make sure everything is 3D (z, x, y), even if starting from a single plane
    if cyto_chan_tile is not None:
        if cyto_chan_tile.ndim == 2:
            cyto_chan_tile = da.expand_dims(cyto_chan_tile, 0)
    if nuclei_chan_tile is not None:
        if nuclei_chan_tile.ndim == 2:
            nuclei_chan_tile = da.expand_dims(nuclei_chan_tile, 0)

    # max project the data
    if max_project:
        if cyto_chan_tile is not None:
            # expand back into 3D stack
            cyto_chan_tile = da.expand_dims(cyto_chan_tile.max(axis=0),0)
        if nuclei_chan_tile is not None:
            nuclei_chan_tile = da.expand_dims(nuclei_chan_tile.max(axis=0), 0)

    # assume cyto and nuclei are the same size
    if cyto_chan_tile is not None:
        n_z, n_x, n_y = cyto_chan_tile.shape
    else:
        n_z, n_x, n_y = nuclei_chan_tile.shape
    # figure out channel settings
    if cyto_chan_tile is None and nuclei_chan_tile is not None:
        # use only nuclei channel
        channel = [0,0]
        # create image of size (z, 1, x, y)
        to_segment = da.expand_dims(nuclei_chan_tile,1)
    elif cyto_chan_tile is not None and nuclei_chan_tile is None:
        # use only cellbody channel
        channel = [0,0]
        # create image of size (z, 1, x, y)
        to_segment = da.expand_dims(cyto_chan_tile, 1)
    else:
        # use both channels by combining them into a single "RGB" image of size (z, 3, x, y)
        channel = [1,2]
        to_segment = da.stack((cyto_chan_tile, nuclei_chan_tile, da.zeros_like(cyto_chan_tile)), axis=0).swapaxes(0,1)

    return channel, to_segment

def segment_cells(tile:TileLoader, cyto_chan_name:Optional[str]='polyT', nuclei_chan_name:Optional[str]=None, use_gpu:bool=False,
                  model_type:str='cyto3', diameter:int=160.,min_size:float=200.,
                  flow_thresh:float=0.4, max_project:Optional[bool]=False,
                  pretrained_model_path:Optional[str]=None) -> np.ndarray:
    """
    Segment cells in the image stack.
    """
    # get registered data for tile
    data, _ = tile.get_registered_data()
    # rechunk so that each chunk is a single z slice, potentially with multiple channels
    n_z, n_chans, n_x, n_y = data.shape

    if cyto_chan_name is not None:
        cyto_chan = da.squeeze(data[:, data.coords['readout_name'] == cyto_chan_name, :, :].data)
    else:
        cyto_chan = None
    if nuclei_chan_name is not None:
        nuclei_chan = da.squeeze(data[:, data.coords['readout_name'] == nuclei_chan_name, :, :].data)
    else:
        nuclei_chan = None
    if cyto_chan is None and nuclei_chan is None:
        raise ValueError("Must provide at least one channel to segment.")
    #print("Segmenting with channels:", cyto_chan_name, nuclei_chan_name)
    #print("Shapes:", cyto_chan.shape, nuclei_chan.shape) 
    cyto_chan = cyto_chan.rechunk((1, n_x, n_y))
    nuclei_chan = nuclei_chan.rechunk((1, n_x, n_y))
    # preprocess data
    if cyto_chan is not None:
        cyto_chan = preprocess_stack_for_segmentation(cyto_chan)
    if nuclei_chan is not None:
        nuclei_chan = preprocess_stack_for_segmentation(nuclei_chan)
    # rearrange data for segmentation
    channel, to_segment = rearrange_data_for_segmentation(cyto_chan, nuclei_chan, max_project=max_project)

    if n_chans == 1:
        to_segment = da.squeeze(to_segment, axis=1)
    if n_z == 1:
        masks = cellpose_segment(to_segment, diameter, flow_thresh, channel, model_type, use_gpu)
        return masks
    else:
        masks = [delayed(cellpose_segment)(to_segment[i], channels=channel, diameter=diameter, min_size=min_size, flow_threshold=flow_thresh,
                                           model_type=model_type, use_gpu=use_gpu, pretrained_model_path=pretrained_model_path) for i in range(n_z)]
        masks = np.squeeze(np.array(da.compute(masks)))
        return np.squeeze(combine_2d_segmentation_masks_into_3d(masks))

@dataclass
class CleanCellsParameters:
    pass

class ExtractAndCleanFeatures(ParallelTileAnalysisTask):
    """
    Task to take in the cell boundaries from each tile and clean them up.
    Creates a graph where each cell is a node, and overlaps are edges. The graph is then refined to assign cells
    to the fov they are closest to (in terms of cnetroid). Then refined to eliminate
    overlapping cells to leave a single cell occupying a given position.

    The output of this will be a geopandas dataframe being saved, rather than
    using MERlin's complicated format.
    """
    def __init__(self, client:dd.Client, output_path:str,
                 masks:List[da.Array], # list of masks, one for each tile
                 global_alignment:SimpleGlobalAlignment,
                 parameters:Optional[CleanCellsParameters]=None):
        super().__init__(client, masks, output_path)
        self.params = parameters
        self.global_align = global_alignment
        self.n_fov = len(masks)
        self.fov_boxes = global_alignment.get_fov_boxes()[:self.n_fov]
        self.all_fovs = np.arange(self.n_fov)
        self.cell_features = {}
        self.temporary_output = os.path.join(self.output_path, "cell_features")
        if not os.path.exists(self.temporary_output):
            os.makedirs(self.temporary_output)

    #def run(self):
    #    self.logger.info("Extracting cell features from masks")
        # extract the cell features from each tile
    #    masks = self.client.scatter(self.tiles)
    #    futures = [self.client.submit(extract_cell_features_for_fov, i, masks[i], self.global_align) for i in range(self.n_fov)]

        # wait for all the futures to finish
    #    self.cell_features = [None] * self.n_fov # pre allocate these so the order can be preserved
    #    while len(futures) > 0:
    #        for i, f in enumerate(futures):
    #            if f.done():
    #                tile_idx, cell_feats = f.result()
    #                self.cell_features[tile_idx] = cell_feats
    #                futures.pop(i)
    #                self.logger.info(f"Finished processing tile {tile_idx}. {len(futures)} tiles remaining.")
    #            elif f.status == 'error':
    #                self.logger.error(f"Error processing tile {tile_idx}. {len(futures)} tiles remaining.")
    #                futures.pop(i)
    #        time.sleep(1)

        #self.cell_features = self.client.gather(futures)
        #self.cell_features = da.compute([delayed(extract_cell_features_for_fov)(i, self.masks[i], self.global_align) for i in range(self.n_fov)])[0]

    def _process_tile(self, tile_idx: int) -> Tuple[int, Future]:
        mask = self.client.scatter(self.tiles[tile_idx])
        return tile_idx, self.client.submit(extract_cell_features_for_fov, tile_idx, mask, self.global_align)

    def _tile_processing_done_callback(self, tile_idx:int, future:dd.Future) -> None:
        self.logger.info(f"Tile {tile_idx} done")
        # store the results
        self.cell_features[tile_idx] = future.result() # type: ignore
        #cell_feats = [x.to_geopandas() for x in self.cell_features[tile_idx]]


    def _postrun_processing(self) -> None:
        self.logger.info("Computing cell overlap graphs")
        intersecting_graphs = [self._get_cell_overlap_graph(i) for i in tqdm(range(self.n_fov))]
        self.logger.info("Computing overlapping cells to remove")
        cleaned_cells_df = self._remove_overlapping_cells(intersecting_graphs)
        self.logger.info("Cleaning cells")
        cleaned_cells = []
        for i in tqdm(range(self.n_fov)):
            cleaned_cells.extend(self._clean_cells(i, cleaned_cells_df))

        # save out the cell features
        self.logger.info("Saving out cleaned cell features")
        cleaned_cells_df = self._convert_to_geopandas(cleaned_cells)
        cleaned_cells_df['id'] = cleaned_cells_df['id'].astype(str)
        cleaned_cells_df.to_parquet(os.path.join(self.temporary_output, 'cell_features.parquet'))


    def _convert_to_geopandas(self, cell_features:List[SpatialFeature]) -> geo.GeoDataFrame:
        """
        Convert the cell features to a geopandas dataframe.
        """
        cell_features = [x.to_geopandas() for x in cell_features]
        return geo.GeoDataFrame(pd.concat(cell_features, ignore_index=True))


    def _get_cell_overlap_graph(self, tile_idx:int) -> List[int]:
        """
        Does some simple cleaning of cells, and then constructs a graph of potential overlaps between cells.
        """
        # get the bounding boxes of the FOVs
        curr_fov_box = self.fov_boxes[tile_idx]
        fov_intersections = sorted([i for i, x in enumerate(self.fov_boxes)
                                    if curr_fov_box.intersects(x)])
        intersecting_fovs = list(self.all_fovs[np.array(fov_intersections)])
        spatial_tree = rtree.index.Index()
        count = 0
        id_to_num = {}

        for i in intersecting_fovs:
            cells = self.cell_features[i]
            cells = simple_clean_cells(cells)
            # progressively updates id_to_num
            spatial_tree, count, id_to_num = construct_tree(cells, spatial_tree, count, id_to_num)

        graph = nx.Graph()
        cells = self.cell_features[tile_idx]
        cells = simple_clean_cells(cells)
        graph = construct_graph(graph, cells, spatial_tree, tile_idx, self.all_fovs, self.fov_boxes)
        return graph

    def _remove_overlapping_cells(self, intersecting_graphs:List[nx.Graph]) -> pd.DataFrame:
        """
        Takes in a graph of cells and removes overlapping cells, returning a dataframe with
        the identities of the cells to keep
        """
        graph = nx.Graph()
        for fov in self.all_fovs:
            sub_graph = intersecting_graphs[fov]
            graph = nx.compose(graph, sub_graph)
        cleaned_cells_df = remove_overlapping_cells(graph)
        return cleaned_cells_df

    def _clean_cells(self, tile_idx:int, cleaned_cells_df:pd.DataFrame) -> List[SpatialFeature]:
        cleaned_cells = cleaned_cells_df[cleaned_cells_df['originalFOV'] == tile_idx]
        cleaned_groups = cleaned_cells.groupby('assignedFOV')
        all_cells_for_fov = []
        for k,g in cleaned_groups:
            cells_to_consider = g['cell_id'].values.tolist()
            features = [x for x in self.cell_features[tile_idx] if x.get_feature_id() in cells_to_consider]
            all_cells_for_fov.extend(features)
        return all_cells_for_fov

def extract_cell_features_for_fov(tile_idx:int, mask:da.Array, global_align:SimpleGlobalAlignment) -> List[SpatialFeature]:
    """
    Extract the cell features for a single fov.
    """
    # get the masks for this fov

    mask = mask.compute()
    z_pos = np.arange(mask.shape[0])
    feature_list = [SpatialFeature.feature_from_label_matrix(
        (mask  == j), tile_idx,
        global_align.fov_to_global_transform(tile_idx),
        z_pos) for j in np.unique(mask) if j != 0]

    # perform some initial filtering
    return feature_list
