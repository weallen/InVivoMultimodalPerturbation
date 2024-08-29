# 
# Functions and classes to process the results of a MERFISH decoding, to filter out and assign molecules
#

from typing import List, Tuple, Optional
from dask.distributed import Client
import pandas as pd
import dask.array as da
from dask.delayed import delayed
import dask.distributed as dd
import pandas as pd
import geopandas as geo
import numpy as np
from dataclasses import dataclass
from skimage import measure
from scipy.spatial import KDTree
import anndata as ad
import os
import shapely

from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree

from collections import defaultdict
from tqdm import tqdm
from pftools.pipeline.core.algorithm import ParallelAnalysisTask 
from pftools.pipeline.processing.globalalign import SimpleGlobalAlignment
from pftools.pipeline.core.codebook import Codebook


def find_containing_polygon(polygons:List[Polygon], tree:STRtree, pt:Point) -> Optional[int]:
    for idx in tree.query(pt):
        if polygons[idx].contains(pt):
            return idx
    return None

#def get_points_for_mol(mol:pd.DataFrame) -> Point:
#    return Point(mol.global_x, mol.global_y)

def assign_molecules_strtree(curr_mols:pd.DataFrame, curr_cells:geo.GeoDataFrame) -> pd.DataFrame:
    # get the molecules for the current cell
    cell_geom = list(curr_cells.geometry)
    if curr_mols.shape[0] > 0:
        # get the points for the molecules
        points = geo.points_from_xy(curr_mols['global_x'], curr_mols['global_y'])
        # create a spatial index for the cells
        tree = STRtree(cell_geom)
        # assign each molecule to a cell
        cell_ids = []
        for i in range(len(points)):
            cell_idx = find_containing_polygon(cell_geom, tree, points[i])
            if cell_idx is not None:
                cell_ids.append(curr_cells.iloc[cell_idx]['id'])
            else:
                cell_ids.append(-1)
        curr_mols = curr_mols.assign(cell_index=cell_ids)
    return curr_mols


class MoleculeAssigner(ParallelAnalysisTask):
    def __init__(self, client:dd.Client, 
                 codebook: Codebook,
                 molecules:pd.DataFrame,
                 cells:geo.GeoDataFrame,
                 global_align:SimpleGlobalAlignment,
                 output_path:str,
                 output_prefix:Optional[str]=None):

        super().__init__(client, output_path)
        self.codebook = codebook
        self.molecules = molecules
        self.cells = cells
        self.global_align = global_align
        self.output_prefix = output_prefix

        # scale values into range X to Y
    def run(self):
        self.logger.info("Assigning molecules to cells")

        #futures = [delayed(self._assign_molecules_to_cells_for_tile)(i, self._get_barcodes_for_tile_with_overlaps(i), 
                                                                     #self.cells[self.cells['fov']==i]) 
                                                                     #for i in self.cells['fov'].unique()]
        #annot_mols = pd.concat(self.client.gather(self.client.compute(futures)), ignore_index=True)
        futures = [self._assign_molecules_to_cells_for_tile(i, self._get_barcodes_for_tile_with_overlaps(i), 
                                                                     self.cells[self.cells['fov']==i]) 
                                                                     for i in tqdm(self.cells['fov'].unique())]
        annot_mols = pd.concat(futures, ignore_index=True)

        self.logger.info("Saving assigned molecules")
        if self.output_prefix is None:
            annot_mols.to_parquet(os.path.join(self.output_path, 'assigned_molecules.parquet'), index=False)
        else:
            annot_mols.to_parquet(os.path.join(self.output_path, self.output_prefix + '_assigned_molecules.parquet'), index=False)

        # convert to scanpy anndata
        self.logger.info("Counting molecules")
        #futures = [delayed(self._count_barcodes_per_cell_for_tile)(i, annot_mols[annot_mols['fov']==i]) for i in annot_mols['fov'].unique()]
        #counts_df = pd.concat(self.client.gather(self.client.compute(futures)), ignore_index=True)
        futures = [self._count_barcodes_per_cell_for_tile(i, annot_mols[annot_mols['fov']==i]) for i in tqdm(annot_mols['fov'].unique())]
        counts_df = pd.concat(futures, ignore_index=True)


        self.logger.info("Saving as scanpy anndata")
        self._save_as_anndata(counts_df)

    def _get_barcodes_for_tile_with_overlaps(self, tile_idx:int) -> pd.DataFrame:
        """
        Get the barcodes for a tile, including barcodes from overlapping tiles just to be extra sure.
        """
        fox_boxes = self.global_align.get_fov_boxes()
        fov_intersections = sorted([i for i,x in enumerate(fox_boxes) if fox_boxes[tile_idx].intersects(x)])
        # get the barcodes for the first intersecting fov
        curr_fov_barcodes = self.molecules[self.molecules['fov']==fov_intersections[0]].copy(deep=True)
        # append molecules for other fields of view
        for fi in fov_intersections:
            if fi != fov_intersections[0]:
                mols = self.molecules[self.molecules['fov']==fi]
                curr_fov_barcodes = pd.concat([curr_fov_barcodes, mols], axis=0)
        return curr_fov_barcodes

    def _assign_molecules_to_cells_for_tile(self, tile_idx:int, tile_barcodes:pd.DataFrame, tile_cells:geo.GeoDataFrame) -> pd.DataFrame:
        tile_cells = tile_cells.reset_index()
        print(f"Found {tile_barcodes.shape[0]} molecules for tile {tile_idx} with {tile_cells.shape[0]} cells")
        if tile_cells.shape[0] > 0:
            z_pos = np.array(tile_cells['z'])
            for i, cell in tqdm(tile_cells.iterrows()):
                # get the molecules for the current cell
                z = z_pos[i]
                curr_mols = tile_barcodes[tile_barcodes['z']==z]
                if curr_mols.shape[0] > 0:
                    idx = np.array([j for (x, y, j) 
                                    in zip(curr_mols['global_x'], curr_mols['global_y'], curr_mols.index) 
                                    if shapely.Point(x, y).within(cell['geometry'])])
                    if len(idx) > 0:
                        tile_barcodes.loc[idx, 'cell_index'] = cell['id']
        return tile_barcodes

    def _count_barcodes_per_cell_for_tile(self, tile_idx:int, molecules:pd.DataFrame) -> pd.DataFrame:
        barcode_count = self.codebook.get_barcode_count()
        barcode_idx = np.arange(barcode_count)
        uniq_cells = list(sorted(molecules['cell_index'].unique()))
        n_cells = len(uniq_cells)
        #print(f"Found {molecules.shape[0]} molecules for tile {tile_idx} with {n_cells} cells")
        bc_counts = np.zeros((n_cells, barcode_count))
        for i, cell_idx in enumerate(uniq_cells):
            curr_cell = molecules[molecules['cell_index']==cell_idx]
            barcode_ids = np.array(curr_cell['barcode_id'].values)
            for j in barcode_idx:
                bc_counts[i,j] = np.sum(barcode_ids == j)
        gene_names = [self.codebook.get_name_for_barcode_index(i) for i in barcode_idx]
        counts_df = pd.DataFrame(bc_counts, columns=gene_names, index=uniq_cells)
        return counts_df
    
    def _save_as_anndata(self, counts_df:pd.DataFrame):
        counts_df = counts_df[counts_df.index != -1] # ignore invalid cells
       # subset to export
        #fields = ["fov", "x", "y", "global_x", "global_y","barcode_id"]
        cell_ids = sorted(list(counts_df.index.unique()))
        uniq_cells = self.cells['id'].unique()
        valid_cells = [i for i in cell_ids if i in uniq_cells]
        self.logger.info(f"Found {len(valid_cells)} valid cells")
        counts_df = counts_df.loc[valid_cells,:]

        adata = ad.AnnData(counts_df.values)
        adata.var_names = counts_df.columns
        adata.obs_names = counts_df.index
        all_df = []
        for cell_id in tqdm(valid_cells):
            curr_df = self.cells[self.cells['id'] == cell_id]
            geometry = curr_df['geometry'].values
            curr_df = curr_df.assign(area = np.mean([g.area for g in geometry]))
            curr_df = curr_df[['fov', 'x', 'y', 'id', 'z','area']]
            all_df.append(curr_df.head(1))
        cells_info = pd.concat(all_df, ignore_index=True)
        cells_info.index = cells_info['id']
        cells_info = cells_info.drop(columns=['id'])
        adata = adata[cells_info.index,:]
        adata.obs = cells_info
        if self.output_prefix is None:
            adata.write_h5ad(os.path.join(self.output_path, 'cellxgene.h5ad'))
        else:
            adata.write_h5ad(os.path.join(self.output_path, self.output_prefix + '_cellxgene.h5ad'))

class FastMoleculeAssigner(MoleculeAssigner):
    """
    Assigns molecules using a spatial tree to speed up the process. 
    """
    def __init__(self, client:dd.Client, 
                 codebook: Codebook,
                 molecules:pd.DataFrame,
                 cells:geo.GeoDataFrame,
                 global_align:SimpleGlobalAlignment,
                 output_path:str,
                 output_prefix:Optional[str]=None):

        super().__init__(client, codebook, molecules, cells, global_align, output_path,output_prefix)
        self.codebook = codebook
        self.molecules = molecules
        self.cells = cells
        self.global_align = global_align
        self.z_pos = np.array(self.cells['z'].unique())

    def run(self):
        self.logger.info("Assigning molecules to cells")
        mol_pos = self.molecules[["x","y", "global_x", "global_y","z","fov","barcode_id"]]
        futures = []
        for i in range(len(self.z_pos)):
            curr_mols = mol_pos[mol_pos['z']==self.z_pos[i]]
            curr_cells = self.cells[self.cells['z']==self.z_pos[i]]
            curr_mols = self.client.scatter(curr_mols)
            curr_cells = self.client.scatter(curr_cells)
            futures.append(self.client.submit(assign_molecules_strtree, curr_mols, curr_cells))
        annot_mols = pd.concat(self.client.gather(futures), ignore_index=True)
        # remove background
        annot_mols = annot_mols[annot_mols['cell_index'] != -1]
        #self.logger.info("Saving assigned molecules")
        #annot_mols.to_parquet(os.path.join(self.output_path, 'assigned_molecules.parquet'), index=False)

        # convert to scanpy anndata
        self.logger.info("Counting molecules")
        #futures = [delayed(self._count_barcodes_per_cell_for_tile)(i, annot_mols[annot_mols['fov']==i]) for i in annot_mols['fov'].unique()]
        #counts_df = pd.concat(self.client.gather(self.client.compute(futures)), ignore_index=True)
        futures = [self._count_barcodes_per_cell_for_tile(i, annot_mols[annot_mols['fov']==i]) for i in tqdm(annot_mols['fov'].unique())]
        counts_df = pd.concat(futures, ignore_index=False)
        # remove duplicates 
        counts_df = counts_df[~counts_df.index.duplicated(keep='first')]

        if self.output_prefix is None:
            counts_df.to_parquet(os.path.join(self.output_path, 'cell_counts.parquet'), index=True)
        else:
            counts_df.to_parquet(os.path.join(self.output_path, self.output_prefix+'_cell_counts.parquet'), index=True)

        # save the counts
        self.logger.info("Saving as scanpy anndata")
        self._save_as_anndata(counts_df)

