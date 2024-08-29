from abc import abstractmethod, ABC
from glob import glob
from typing import List
import pandas as pd
import dask.array as da
from dask_image.imread import imread as dask_imread
from pftools.pipeline.export.zarr_export import make_tiff_file_structure_df, validate_tiff_prefixes_and_range
import zarr
from tqdm import tqdm
import os

class ExperimentData(ABC):
    """
    Class that wraps an experiment's data. This could be a Zarr array, a directory of tiffs, etc.
    Has a database of all the tiles, and can return tiles by fov, round, etc.

    Assumes that each file is uniquely marked by prefix, fov, and round.
    Each file is a zstack of images, with the first dimension being the z dimension.n
    """
    def __init__(self, base_name:str):
        self.base_name = base_name
        self.db = pd.DataFrame()
        self.nfovs = -1
        self.prefixes = []
        self.pos = None

    @abstractmethod
    def _load_pos(self) -> pd.DataFrame:
        pass

    def get_positions(self) -> pd.DataFrame:
        return self.pos

    def get_prefixes(self) -> List[str]:
        return self.prefixes
        
    def get_nfovs(self) -> int:
        return int(len(self.nfovs))

    def get_fovs(self) -> List[int]:
        return sorted(list(self.db.fov.unique()))

    def get_data_for_all_tiles(self, prefix:str) -> List[List[da.Array]]:
        """
        Return a list of lists of dask arrays. The outer list is the rounds, the inner list is the fovs.
        """
        if prefix not in self.prefixes:
            raise ValueError(f"Prefix {prefix} not found in database")
        curr_df = self.db[self.db['prefix'] == prefix]
        fovs = []
        for fov in range(self.nfovs):
            curr_df_subset = curr_df[curr_df['fov'] == fov]
            rounds = []
            for round in curr_df_subset['round'].unique():
                rounds.append(self.get_data_by_fov(prefix, round))
            fovs.append(rounds)
        return fovs
    
    def get_data_by_fov(self, prefix:str, fov: int) -> List[da.Array]:
        """
        Return a list of dask arrays for a particular fov
        """
        if len(self.db[(self.db['prefix'] == prefix) & (self.db['fov'] == fov)]) == 0:
            raise ValueError(f"Database is incomplete for prefix {prefix}, fov {fov}")
        data = []
        rounds = sorted(self.db[(self.db['prefix'] == prefix) & (self.db['fov'] == fov)]['round'].unique().astype(int))
        for round in rounds:
            data.append(self.get_data_by_round_and_fov(prefix, round, fov))
        return data
    
    @abstractmethod
    def get_data_by_round_and_fov(self, prefix:str, round: int, fov: int) -> da.Array:
        """
        Return just the single image stack for a particular round. 
        Implementation depends on backing store. 
        """
        pass
    
    @abstractmethod
    def _generate_database(self):
        pass
    


class ExperimentDataZarr(ExperimentData):
    """
    Experiment data backed by Zarr array
    """
    def __init__(self, base_name:str, force_rebuild:bool=False):
        """
        base_name is path to the zarr array
        """
        super().__init__(base_name)
        print("Loading",base_name)
        if base_name.endswith('.zarr') or base_name.endswith(".zarr/"):
            self.zarr = zarr.open(base_name)
            self.readonly = False
        elif base_name.endswith('.zip') or base_name.endswith(".zip/"):
            print("Zip file detected, opening as ZipStore")
            if base_name[-1] == '/':
                # remove trailing slash if a zipfile
                base_name = base_name[:-1]
            self.zarr = zarr.open(base_name)
            self.readonly = True
        if hasattr(self.zarr, 'attrs'):
            if 'db' in self.zarr.attrs and not force_rebuild:
                print("Loading database for experiment", base_name)
                # metadata is already consolidated
                #self._zarr = zarr.open_consolidated(base_name)
                self.db = pd.DataFrame(self.zarr.attrs['db'])
                self.nfovs = len(self.db['fov'].unique())
                self.prefixes = list(self.db['prefix'].unique())
            else:
                print("Generating database for experiment", base_name)
                self._generate_database()
                #zarr.consolidate_metadata(self._zarr)
        else:
            print("Generating database for expeirment", base_name)
            self._generate_database()
        self.pos = self._load_pos()

    def _load_pos(self) -> pd.DataFrame:
        if 'pos' in self.zarr:
            df = pd.DataFrame(self.zarr['pos'], columns=['x','y'])
            return df
        else:
            return None
    
    def get_data_by_round_and_fov(self, prefix:str, round: int, fov: int) -> da.Array:
        """
        Return just the single image stack for a particular round
        """
        if len(self.db[(self.db['prefix'] == prefix) & (self.db['round'] == round) & (self.db['fov'] == fov)]) != 1:
            raise ValueError(f"Database is incomplete for prefix {prefix}, round {round}, fov {fov}")
        return da.from_zarr(self.zarr[f"/{prefix}/fov_{fov}/round_{round}"])
    

    def _generate_database(self):
        root = self.zarr
        combinations = []  # To store the combinations of prefix, fov, and round
        print("Found the following groups in the Zarr array:", list(root.group_keys()))
        # Iterate through each item in the Zarr group
        for prefix, prefix_group in root.groups():
            if prefix != "pos":
                print(prefix)
                for fov, fov_group in tqdm(prefix_group.groups()):
                    for round_ in fov_group.array_keys():
                        # Extract N and M from fov and round_ respectively
                        fov_number = int(fov.split('_')[1])
                        round_number = int(round_.split('_')[1])
                        # Append the combination to the list
                        combinations.append({'prefix': prefix, 'fov': fov_number, 'round': round_number})
                
        # Convert the list of dictionaries to a DataFrame
        self.db = pd.DataFrame(combinations).sort_values(by=['prefix', 'fov', 'round'])
#        self._nfovs = len(self._db['fov'].unique())
#        self._prefixes = self._db['prefix'].unique()
        if not self.readonly:    
            self.zarr.attrs.update({'db': self.db.to_dict()})
        #print("Consolidating metadata...")
        #zarr.consolidate_metadata(self._base_name)


class ExperimentDataFS(ExperimentData):
    """
    Experiment data back by file system.
    """
    def __init__(self, base_name:str, force_rebuild:bool=False):
        super().__init__(base_name)
        if os.path.exists(os.path.join(base_name, "db.csv")) and not force_rebuild:
            print("Loading database for experiment", base_name)
            self.db = pd.read_csv(os.path.join(base_name, "db.csv"))
            self.nfovs = len(self.db['fov'].unique())
            self.prefixes = self.db['prefix'].unique()
        else:
            print("Generating database for experiment", base_name)
            self._generate_database()
            self.db.to_csv(os.path.join(base_name, "db.csv"))
        self.base_name = base_name
        self.pos = self._load_pos() 
        self.pos.columns = ['x','y']
    def _load_pos(self) -> pd.DataFrame:
        if os.path.exists(os.path.join(self.base_name, "tiled_pos.csv")):
            print("Loading positions from tiled_pos.csv") 
            return pd.read_csv(os.path.join(self.base_name, "tiled_pos.csv"))
        elif os.path.exists(os.path.join(self.base_name, "tiled_pos.txt")):
            print("Loading positions from tiled_pos.txt")
            return pd.read_csv(os.path.join(self.base_name, "tiled_pos.txt"))

        elif os.path.exists(os.path.join(self.base_name, "pos_tiled.csv")):
            print("Loading positions from pos_tiled.csv")
            return pd.read_csv(os.path.join(self.base_name, "pos_tiled.csv"))
        elif os.path.exists(os.path.join(self.base_name, "pos_tiled.txt")):
            print("Loading positions from pos_tiled.txt")
            return pd.read_csv(os.path.join(self.base_name, "pos_tiled.txt"))
        else:
            return None
        
    def get_data_by_round_and_fov(self, prefix:str, round: int, fov: int) -> da.Array:
        """
        Return just the single image stack for a particular round
        """
        return dask_imread(self.db[(self.db['prefix'] == prefix) & (self.db['round'] == round) & (self.db['fov'] == fov)]['filename'].values[0])

    def _generate_database(self):
        print("Listing TIFFs")
        tiffs = glob(os.path.join(self.base_name, "*.tif"))
        print("Making TIFF file structure")
        tiff_df = make_tiff_file_structure_df(tiffs)
        tiff_df['fov'] = tiff_df['fovs'].astype(int)
        tiff_df['round'] = tiff_df['rounds'].astype(int)
        if not validate_tiff_prefixes_and_range(tiff_df):
            raise ValueError("Invalid tiff structure")
        tiff_df['prefix'] = [i.split("/")[-1] for i in tiff_df['prefix']]
        tiff_df = tiff_df.loc[:, ['prefix','fov','round','filename']]
        self.db = tiff_df
        self.nfovs = len(tiff_df['fov'].unique())
        self.prefixes = tiff_df['prefix'].unique()

    
