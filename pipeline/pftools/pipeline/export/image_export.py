import dask.distributed as dd
import dask.array as da
from distributed import Future
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
from typing import List, Dict
from scipy.spatial import KDTree
import geopandas as geo
import pandas as pd
from tqdm import tqdm
import zarr
import os
from skimage.measure import regionprops
from skimage.transform import downscale_local_mean
from numcodecs import Blosc
from skimage.exposure import equalize_adapthist
from dask.delayed import delayed
from pftools.pipeline.core.dataorganization import DataOrganization
from pftools.pipeline.core.algorithm import ParallelTileAnalysisTask
from pftools.pipeline.processing.globalalign import SimpleGlobalAlignment
from pftools.pipeline.core.tile import TileLoader
from pftools.pipeline.core.microscope import MicroscopeInformation
from pftools.pipeline.util.cellpose_segment import preprocess_stack_for_segmentation

@dataclass
class ImageExportParameters:
    downsample_factor:int = 2
    crop_width:int = 256
    crop_height:int = 256

class ImageExporter(ParallelTileAnalysisTask):
    """
    Task to take in the cell boundaries from each tile and clean them up.
    Creates a graph where each cell is a node, and overlaps are edges. The graph is then refined to assign cells
    to the fov they are closest to (in terms of cnetroid). Then refined to eliminate
    overlapping cells to leave a single cell occupying a given position.

    The output of this will be a geopandas dataframe being saved, rather than
    using MERlin's complicated format.
    """
    def __init__(self, client:dd.Client,
                 tiles:List[TileLoader], # list of data, one for each tile
                 masks:List[da.Array], # list of masks, one for each tile
                 cell_feats: geo.GeoDataFrame, # shapes for each cell
                 global_alignment:SimpleGlobalAlignment,
                 microscope:MicroscopeInformation,
                 dataorg:DataOrganization,
                 output_path:str,
                 parameters:Optional[ImageExportParameters]=None):
        super().__init__(client, tiles, output_path)
        if parameters is None:
            parameters = ImageExportParameters()
        self.params = parameters
        self.global_align = global_alignment
        self.microscope = microscope
        self.data_org = dataorg
        self.masks = masks
        self.cell_feats = cell_feats
        self.cell_feats = self.cell_feats.sort_values('fov') # sort by FOV first
        self.n_fov = len(cell_feats.fov.unique())
        #self.fov_boxes = global_alignment.get_fov_boxes()
        #self.all_fovs = np.arange(self.n_fov)
        self.output_path = output_path
        self._setup_output_file(output_path)

    def __del__(self):
        if hasattr(self, 'output_file_store'):
            self.output_file_store.close()

    def _setup_output_file(self, output_path:str):
        self.logger.info(f"Opening output in {output_path}")
        fname = os.path.join(output_path, 'cell_images.zarr')
        # if the file exists, open it in append mode
        if os.path.exists(fname):
            if fname.endswith('.zip'):
                self.output_file_store = zarr.ZipStore(fname, mode='a')
                self.output_file = zarr.group(store=self.output_file_store)
            else:
                self.output_file = zarr.open(fname, mode='a')
        else:
            # otherwise create a new file
            if fname.endswith('.zip'):
                self.output_file_store = zarr.ZipStore(fname, mode='w')
                self.output_file = zarr.group(store=self.output_file_store)
            else:
                self.output_file = zarr.open(fname, mode='w')

            # set the attributes
            self.output_file.attrs['downsample_factor'] = self.params.downsample_factor
            self.output_file.attrs['crop_width'] = self.params.crop_width
            self.output_file.attrs['crop_height'] = self.params.crop_height

            # save channels
            data_info_df = self.data_org.get_data_info_df()
            self.output_file.attrs['readout_names'] = list(data_info_df.readout_name.unique())
            self.output_file.attrs['bit_names'] = list(data_info_df.name.unique())

    def _process_tile(self, tile_idx: int) -> Tuple[int, Future]:
        tile = self.tiles[tile_idx]
        mask = self.masks[tile_idx]
        if not f"fov_{tile_idx}" in self.output_file:
            return tile_idx, self.client.submit(export_cells_for_fov, tile_idx, tile, mask, self.params.crop_width, self.params.crop_height, self.params.downsample_factor)
        else:
            self.logger.info(f"Tile {tile_idx} already exported.")
            return tile_idx, None

    def _tile_processing_done_callback(self, tile_idx:int, future:dd.Future) -> None:
        self.logger.info(f"Tile {tile_idx} done. Matching cells with cell features.")
        # store the results
        seg_temp, cell_df_temp = future.result()
        seg = seg_temp.copy()
        cell_df = cell_df_temp.copy()
        del seg_temp
        del cell_df_temp

        #self.logger.info(f"Found {seg.shape[0]} cells in tile {tile_idx}.")
        # match the cells
        curr_cell_feats = self.cell_feats[self.cell_feats.fov == tile_idx]
        curr_cell_feats.loc[:,'centroid_x'] = curr_cell_feats['geometry'].centroid.x
        curr_cell_feats.loc[:,'centroid_y'] = curr_cell_feats['geometry'].centroid.y
        cell_locs = (curr_cell_feats[['centroid_x','centroid_y']].values - self.global_align.fov_offsets[tile_idx])/self.microscope.microns_per_pixel

        mask_tree = KDTree(cell_df[["y","x"]].values)
        cell_tree = KDTree(cell_locs)
        cell_tree_idx = mask_tree.query_ball_tree(cell_tree, r=0.5)
        ids = curr_cell_feats['id'].values
        # assign the cell ids to the cell_df based on the nearest neighbor (within half a micron)
        cell_df['cellxgene_cell_id'] = [ids[i[0]] if len(i) > 0 else -1 for i in cell_tree_idx]

        # subset to only cells that have a match and save out
        self.logger.info(f"Found {len(cell_df['cell_id'].unique())} cells with {len(cell_df[cell_df.cellxgene_cell_id != -1]['cell_id'].unique())} matches, out of {len(curr_cell_feats['id'].unique())} total cells.")
        cells_to_keep = cell_df[cell_df.cellxgene_cell_id != -1]

        # create a group for this field of view
        compressor = Blosc(cname='lz4', clevel=9, shuffle=Blosc.BITSHUFFLE)
        with self.output_file.create_group(f'fov_{tile_idx}') as curr_fov:
            if self.output_path.endswith('.zip'): # can't write in parallel for zip
                for c in cells_to_keep['cell_id'].unique():
                    _save_cell_img(curr_fov, seg[c-1], cells_to_keep[cells_to_keep['cell_id']==c].cellxgene_cell_id.values[0], compressor)
            else:
                da.compute([delayed(_save_cell_img)(curr_fov, 
                                                seg[c-1], 
                                                cells_to_keep[cells_to_keep['cell_id']==c].cellxgene_cell_id.values[0], compressor) 
                            for c in cells_to_keep['cell_id'].unique()])
                    # there will be multiple z planes for each cell, but just take the first unique cell id

        # save the results
    def _postrun_processing(self) -> None:
        pass

def _save_cell_img(curr_fov, data, curr_id, compressor):
            # there will be multiple z planes for each cell, but just take the first unique cell id
            try:
                curr_fov.create_dataset(str(curr_id),
                                    data=data,
                                    shape=data.shape,
                                    chunks=(data.shape[0],data.shape[1],data.shape[2]),
                                    compressor=compressor,
                                    dtype=data.dtype)
            except Exception as e:
                print(f"Error saving cell assigned to {curr_id}.")
                print(e)

 
# find middle z plane for each cell
from skimage.measure import regionprops
def compute_cell_info_df(seg:np.ndarray) -> pd.DataFrame:
    uniq_cells = np.unique(seg)
    cell_z = []
    cell_area = []
    cell_id = []
    cell_x = []
    cell_y = []
    n_z, n_x, n_y = seg.shape
    for i in uniq_cells:
        if i > 0:
            curr_cell_z = []
            curr_cell_id = []
            curr_cell_area = []
            curr_cell_x = []
            curr_cell_y = []
            for z in range(n_z):
                rp = regionprops((seg[z]==i).astype(np.uint8))
                if len(rp) > 0:
                    x, y = rp[0].centroid
                    curr_cell_area.append(rp[0].area)
                    curr_cell_z.append(z)
                    curr_cell_id.append(i)
                    curr_cell_x.append(x)
                    curr_cell_y.append(y)
            cell_z.extend(curr_cell_z)
            cell_id.extend(curr_cell_id)
            cell_area.extend(curr_cell_area)
            cell_x.extend(curr_cell_x)
            cell_y.extend(curr_cell_y)
    return pd.DataFrame({'cell_id': cell_id, 'z_plane':cell_z, 'area':cell_area, 'x':cell_x, 'y':cell_y})

def find_max_area_z(z_df:pd.DataFrame) -> pd.DataFrame:
    cell_ids = []
    zs = []
    xs = []
    ys = []
    for cell_id in z_df.cell_id.unique():
        curr_z_df = z_df[z_df.cell_id == cell_id]
        z_planes = curr_z_df.z_plane.values
        max_area = np.argmax(curr_z_df.area.values)
        cell_ids.append(cell_id)
        zs.append(z_planes[max_area])
        xs.append(curr_z_df.x.values[max_area])
        ys.append(curr_z_df.y.values[max_area])
    return pd.DataFrame({'cell_id':cell_ids, 'z_plane':zs, 'x':xs, 'y':ys})

def crop_and_pad_image(image, coord, x_width, y_width):
    crop_width = 2 * x_width
    crop_height = 2 * y_width

    # Determine the crop bounds, ensuring they do not go beyond the image boundaries
    left = max(coord[0] - x_width, 0)
    right = min(coord[0] + x_width, image.shape[1])
    top = max(coord[1] - y_width, 0)
    bottom = min(coord[1] + y_width, image.shape[0])

    # Crop the image
    cropped_image = image[top:bottom, left:right]

    # Calculate the padding needed to achieve the desired output dimensions
    pad_left = x_width - (coord[0] - left)
    pad_right = crop_width - (right - left) - pad_left
    pad_top = y_width - (coord[1] - top)
    pad_bottom = crop_height - (bottom - top) - pad_top

    # Apply padding
    padded_image = np.pad(cropped_image, ((pad_top, pad_bottom), (pad_left, pad_right)), 'constant', constant_values=0)

    return padded_image

def clip_high_intensity(image:np.ndarray, clip_value:float=10):
    mean = image.mean()
    std = image.std()
    thresh = mean + clip_value * std
    return np.clip(image, 0, thresh)

def _clip_and_equalize(image:np.ndarray):
    image = clip_high_intensity(image)
    return equalize_adapthist(image/image.max(), clip_limit=0.01, kernel_size=256, nbins=256)
    
# crop directly from the segmentation
def segment_cells(tile: da.Array, seg: np.ndarray, x_width:int=128, y_width:int=128, downsample_factor=2):
    uniq_cells = np.unique(seg)
    cell_df = compute_cell_info_df(seg)

    max_z = find_max_area_z(cell_df)
    n_z, n_feat, n_x, n_y = tile.shape
    cell_imgs = np.zeros((len(uniq_cells)-1, n_feat, x_width*2 // downsample_factor, y_width*2 // downsample_factor), dtype=np.float32)
    cell_ids = []
    # first clip and adaptively histogram equalize channels
    tile = tile.astype(np.float32)
    for i in range(n_z):
        tile[i,:] = np.stack(da.compute([delayed(_clip_and_equalize)(tile[i,j]) for j in range(n_feat)])[0])

        #for j in range(n_feat):
        #    tile[i,j] = clip_high_intensity(tile[i,j])
        #    tile[i,j] = equalize_adapthist(tile[i,j]/tile[i,j].max(), clip_limit=0.01, kernel_size=256, nbins=256)

    for n,i in enumerate(uniq_cells):
        # skip 0 which is the background
        if i > 0:
            # identify the z plane to use
            curr_z = max_z[max_z.cell_id == i].z_plane.values[0]

            # identify the centroid of the mask for the crop
            x_pos, y_pos = regionprops((seg[curr_z] == i).astype(np.uint8))[0].centroid
            x_pos, y_pos = int(x_pos), int(y_pos)
            # Crop the cell from each channel

            for j in range(n_feat):
                # get the data channels for that z plane and feature
                channel_img = tile[curr_z, j].copy()
                # mask the image
                channel_img[seg[curr_z] != i] = 0
                # add a crop
                curr_crop = crop_and_pad_image(channel_img, (y_pos, x_pos), x_width, y_width)
                #if curr_crop.shape[0] == 256 and curr_crop.shape[1] == 256:
                if downsample_factor > 1:
                    cell_imgs[n-1,j] = downscale_local_mean(curr_crop, (downsample_factor, downsample_factor))
                else:
                    cell_imgs[n-1,j] = curr_crop
            cell_ids.append(i)
    return cell_imgs, cell_df

def export_cells_for_fov(tile_idx:int, tile:TileLoader, mask:np.ndarray, x_width:int=128, y_width:int=128, downsample_factor:int=2) -> Dict:
    """
    Extracts the cell features from a single tile mask
    """
    # get the cell info

    # get the cell features
    tile, _ = tile.get_registered_data()
    tile = tile.compute()
    mask = mask.compute()
    segs, cell_df = segment_cells(tile.data, mask, x_width=x_width, y_width=y_width, downsample_factor=downsample_factor)

    return segs, cell_df
