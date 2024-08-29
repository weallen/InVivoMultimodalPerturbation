# Adapted from MERlin:
# https://raw.githubusercontent.com/ZhuangLab/MERlin/master/merlin/data/dataorganization.py
from __future__ import annotations
import os
import re
from typing import List, Optional
from typing import Tuple
import pandas
import numpy as np
from tqdm import tqdm
import dask_image
import dask.array as da
from dask_image.imread import imread
from pftools.pipeline.core.dataset import ExperimentData

class DataFormatException(Exception):
    pass

def _parse_list(inputString: str, dtype=float):
    if ',' in inputString:
        return np.fromstring(inputString.strip('[] '), dtype=dtype, sep=',')
    else:
        return np.fromstring(inputString.strip('[] '), dtype=dtype, sep=' ')


def _parse_int_list(inputString: str):
    return _parse_list(inputString, dtype=int)



class DataOrganization(object):
    """
    Loads the data organization for an data set, which is the mapping between the data channels and the image files,
    as well as a specification of the fiducial image channels.

    ExperimentData contains the overall structure of all files for the experiment.
    DataOrganization loads in a file specifying which z planes on which rounds correspond to which channels. 

    max_fov is to only load a subset of the data. 
    Each image in the experiment is specified by a unique tuple (imageType, fov, imagingRound, zPosition)
    """
    def __init__(self, file_path:str, max_fov:Optional[int]=None):
        self.max_fov = max_fov
        self.data = pandas.read_csv(
            file_path,
            converters={'frame': _parse_int_list, 'zPos': _parse_list})
        self.data['readoutName'] = self.data['readoutName'].str.strip()

        stringColumns = ['readoutName', 'channelName', 'imageType', 'fiducialImageType']
        self.data[stringColumns] = self.data[stringColumns].astype('str')
        # subset to used part of filemap based on data org
        self.dorg = self.get_data_info_df()

    # methods for getting general information about the data organization
    def get_dorg_df(self) -> pandas.DataFrame:
        return self.dorg

    def get_data_info_df(self) -> pandas.DataFrame:
        """
        Get a dataframe with one row for each image in a single FOV.
        Assumes that fiducials are stored as a single plane in the same image file as the data.
        Enumerates all tuples of (imageType, fov, imagingRound, zPosition)
        """
        # convert the filenames to indices into the list of filenames
        #filenames = self.get_data_filenames_for_fov(0)
        #filename_map = {}
        #filename_idx = 0
        #for f in filenames:
        #    if f not in filename_map:
        #        filename_map[f] = filename_idx
        #        filename_idx += 1

        zpos = self.get_z_positions()
        per_chan_info = {}
        per_chan_info['color'] = []
        per_chan_info['name'] = []
        per_chan_info['z'] = []
        per_chan_info['readout_name'] = []
        per_chan_info['round'] = []
        per_chan_info['frame_idx'] = []
        per_chan_info['image_type'] = []
        per_chan_info['image_idx'] = []
        image_idx = 0
        for i in self.get_data_channels():
            rnd = self.get_data_channel_round(i)
            c = self.get_data_channel_color(i)
            n = self.get_data_channel_name(i)
            r = self.get_data_channel_readout_name(i)
            it = self.get_data_channel_image_type(i)
            for z in zpos:
                idx = self.get_image_frame_index(i, z)
                per_chan_info['frame_idx'].append(idx)
                per_chan_info['color'].append(c)
                per_chan_info['name'].append(n)
                per_chan_info['readout_name'].append(r)
                per_chan_info['z'].append(float(z))
                per_chan_info['round'].append(rnd)
                per_chan_info['image_type'].append(it)
                per_chan_info['image_idx'].append(image_idx)#filename_map[self.get_image_filename(i, 0)])
            image_idx += 1
        return pandas.DataFrame(per_chan_info)

    def get_fiducial_info_df(self, per_channel=True) -> pandas.DataFrame:
        per_chan_info = {}
        per_chan_info['frame_idx'] = []
        per_chan_info['round'] = []
        per_chan_info['image_idx'] = []
        per_chan_info['image_type'] = []
        per_chan_info['readout_name'] = []
        #filename_map = {}
        #filename_idx = 0
        #for f in self.get_fiducial_filenames_for_fov(0):
        #    if f not in filename_map:
        #        filename_map[f] = filename_idx
        #        filename_idx += 1            
        image_idx = 0
        for i in self.get_data_channels():
            fiducial_idx = self.get_fiducial_frame_index(i)
            it = self.get_data_channel_image_type(i)
            r = self.get_data_channel_readout_name(i)
            rnd = self.get_data_channel_round(i)
            per_chan_info['image_idx'].append(image_idx)#filename_map[self.get_fiducial_filename(i, 0)])
            per_chan_info['frame_idx'].append(fiducial_idx)
            per_chan_info['image_type'].append(it)
            per_chan_info['readout_name'].append(r)
            per_chan_info['round'].append(rnd)
            image_idx += 1
        if per_channel:
            return pandas.DataFrame(per_chan_info)
        else:
            # subset to the first frame of each channel
            return pandas.DataFrame(per_chan_info).drop_duplicates(subset=['image_idx'])

    def get_data_channels(self) -> np.array:
        """Get the data channels for the MERFISH data set."""
        return np.array(self.data.index)

    def get_data_channel_readout_name(self, dataChannelIndex: int) -> str:
        """Get the name for the data channel with the specified index."""
        return self.data.iloc[dataChannelIndex]['readoutName']

    def get_data_channel_image_type(self, dataChannelIndex: int) -> str:
        """Get the image type for the data channel with the specified index."""
        return self.data.iloc[dataChannelIndex]['imageType']

    def get_data_channel_round(self, dataChannelIndex: int) -> int:
        """Get the imaging round for the data channel with the specified index."""
        return self.data.iloc[dataChannelIndex]['imagingRound']

    def get_data_channel_name(self, dataChannelIndex: int) -> str:
        """Get the name for the data channel with the specified index."""
        return self.data.iloc[dataChannelIndex]['channelName']

    def get_data_channel_index(self, dataChannelName: str) -> int:
        """Get the index for the data channel with the specified name."""
        return self.data[self.data['channelName'].apply(
            lambda x: str(x).lower()) == str(dataChannelName).lower()]\
            .index.values.tolist()[0]

    def get_data_channel_color(self, dataChannel: int) -> str:
        """Get the color used for measuring the specified data channel."""
        return str(self.data.at[dataChannel, 'color'])

    def get_data_channel_for_bit(self, bitName: str) -> int:
        """Get the data channel associated with the specified bit."""
        return self.data[self.data['readoutName'] ==
                         bitName].index.values.item()

    def get_data_channel_with_name(self, channelName: str) -> int:
        """Get the data channel associated with a gene name."""
        return self.data[self.data['channelName'] ==
                         channelName].index.values.item()

    def get_fiducial_filename(self, dataChannel: int, fov: int) -> str:
        """Get the path for the image file that contains the fiducial"""
        imageType = self.data.loc[dataChannel, 'fiducialImageType']
        imagingRound = \
            self.data.loc[dataChannel, 'fiducialImagingRound']
        return self._get_image_path(imageType, fov, imagingRound)

    def get_fiducial_frame_index(self, dataChannel: int) -> int:
        """Get the index of the frame containing the fiducial image
        for the specified data channel."""
        return self.data.iloc[dataChannel]['fiducialFrame']

    def get_image_frame_index(self, dataChannel: int, zPosition: float) -> int:
        """Get the index of the frame containing the image
        for the specified data channel and z position."""
        channelInfo = self.data.iloc[dataChannel]
        channelZ = channelInfo['zPos']
        if isinstance(channelZ, np.ndarray):
            zIndex = np.where(channelZ == zPosition)[0]
            if len(zIndex) == 0:
                raise Exception('Requested z position not found. Position ' +
                                'z=%0.2f not found for channel %i'
                                % (zPosition, dataChannel))
            else:
                frameIndex = zIndex[0]
        else:
            frameIndex = 0

        frames = channelInfo['frame']
        if isinstance(frames, np.ndarray):
            frame = frames[frameIndex]
        else:
            frame = frames

        return frame

    def get_z_positions(self) -> List[float]:
        """Get the z positions present in this data organization."""
        return sorted(np.unique([y for x in self.data['zPos'] for y in x]))

   
class DataOrganizationOrig(object):

    """
    A class to specify the organization of raw images in the original
    image files.
    """

    def __init__(self, filePath: str, dataPath:str, fileMap: str = None, max_fov:int=None):
        """
        Create a new DataOrganization for the data in the specified data set.

        Raises:
            InputDataError: If the set of raw data is incomplete or the
                    format of the raw data deviates from expectations.
        """
        self._data_path = dataPath 
        self._max_fov = max_fov
        self.data = pandas.read_csv(
            filePath,
            converters={'frame': _parse_int_list, 'zPos': _parse_list})
        self.data['readoutName'] = self.data['readoutName'].str.strip()

        stringColumns = ['readoutName', 'channelName', 'imageType', 
                         'imageRegExp', 'fiducialImageType', 'fiducialRegExp']
        self.data[stringColumns] = self.data[stringColumns].astype('str')
        self.fileMap = fileMap
        # subset to used part of filemap based on data org
        df = self.get_data_info_df()
        if self._max_fov is None:
            self._max_fov = df['fov'].max()
        self.fileMap = self.fileMap[self.fileMap.imageType.isin(self.data.imageType.unique())]
        self.fileMap = self.fileMap[self.fileMap['fov'] < self._max_fov]
        self.fileMap = self.fileMap[self.fileMap['imagingRound'].isin(self.data.imagingRound.unique())]
        
        # subset to actually used part of fileMap if max_fov is set
        # if fileMap isn't cached, then load it from the file
        #if fileMapPath is None:
        #    self._map_image_files()
        #else:
        #    self.fileMap = pandas.read_csv(fileMapPath)

    def save_filemap_df(self, filemap_path:str) -> None:
        """
        Save the filemap dataframe to a CSV file
        """
        self.fileMap.to_csv(filemap_path, index=False)

    def get_data_filenames_for_fov(self, fov:int, per_channel=False) -> List[str]:
        """
        Get the filenames for a single FOV.
        """
        fnames = [self.get_image_filename(i, fov) for i in self.get_data_channels()]
        if per_channel:
            return fnames
        else:
            return list(dict.fromkeys(fnames))

    def get_data_info_df(self) -> pandas.DataFrame:
        """
        Get a dataframe with one row for each image in a single FOV.
        Assumes that fiducials are stored as a single plane in the same image file as the data.
        """
        # convert the filenames to indices into the list of filenames
        filenames = self.get_data_filenames_for_fov(0)
        filename_map = {}
        filename_idx = 0
        for f in filenames:
            if f not in filename_map:
                filename_map[f] = filename_idx
                filename_idx += 1

        zpos = self.get_z_positions()
        per_chan_info = {}
        per_chan_info['color'] = []
        per_chan_info['name'] = []
        per_chan_info['z'] = []
        per_chan_info['readout_name'] = []
        per_chan_info['round'] = []
        per_chan_info['frame_idx'] = []
        per_chan_info['image_type'] = []
        per_chan_info['image_idx'] = []
        for i in self.get_data_channels():
            rnd = self.get_data_channel_round(i)
            c = self.get_data_channel_color(i)
            n = self.get_data_channel_name(i)
            r = self.get_data_channel_readout_name(i)
            it = self.get_data_channel_image_type(i)
            for z in zpos:
                idx = self.get_image_frame_index(i, z)
                per_chan_info['frame_idx'].append(idx)
                per_chan_info['color'].append(c)
                per_chan_info['name'].append(n)
                per_chan_info['readout_name'].append(r)
                per_chan_info['z'].append(float(z))
                per_chan_info['round'].append(rnd)
                per_chan_info['image_type'].append(it)
                per_chan_info['image_idx'].append(filename_map[self.get_image_filename(i, 0)])
        return pandas.DataFrame(per_chan_info)

    def get_fiducial_filenames_for_fov(self, fov:int, per_channel=True) -> List[str]:
        fnames = [self.get_fiducial_filename(i, fov) for i in self.get_data_channels()]
        if per_channel:
            return fnames
        else:
            return list(dict.fromkeys(fnames))


    def get_fiducial_info_df(self, per_channel=True) -> pandas.DataFrame:
        per_chan_info = {}
        per_chan_info['frame_idx'] = []
        per_chan_info['round'] = []
        per_chan_info['image_idx'] = []
        filename_map = {}
        filename_idx = 0
        for f in self.get_fiducial_filenames_for_fov(0):
            if f not in filename_map:
                filename_map[f] = filename_idx
                filename_idx += 1            
        for i in self.get_data_channels():
            fiducial_idx = self.get_fiducial_frame_index(i)
            per_chan_info['image_idx'].append(filename_map[self.get_fiducial_filename(i, 0)])
            per_chan_info['frame_idx'].append(fiducial_idx)
            per_chan_info['round'].append(i)
        if per_channel:
            return pandas.DataFrame(per_chan_info)
        else:
            # subset to the first frame of each channel
            return pandas.DataFrame(per_chan_info).drop_duplicates(subset=['image_idx'])

    def get_data_channels(self) -> np.array:
        """Get the data channels for the MERFISH data set.

        Returns:
            A list of the data channel indexes
        """
        return np.array(self.data.index)

    def get_data_channel_readout_name(self, dataChannelIndex: int) -> str:
        """Get the name for the data channel with the specified index.

        Args:
            dataChannelIndex: The index of the data channel
        Returns:
            The name of the specified data channel
        """
        return self.data.iloc[dataChannelIndex]['readoutName']

    def get_data_channel_image_type(self, dataChannelIndex: int) -> str:
        """Get the image type for the data channel with the specified index.

        Args:
            dataChannelIndex: The index of the data channel
        Returns:
            The image type of the specified data channel
        """
        return self.data.iloc[dataChannelIndex]['imageType']

    def get_data_channel_round(self, dataChannelIndex: int) -> int:
        """Get the imaging round for the data channel with the specified index.

        Args:
            dataChannelIndex: The index of the data channel
        Returns:
            The imaging round of the specified data channel
        """
        return self.data.iloc[dataChannelIndex]['imagingRound']

    def get_data_channel_name(self, dataChannelIndex: int) -> str:
        """Get the name for the data channel with the specified index.

        Args:
            dataChannelIndex: The index of the data channel
        Returns:
            The name of the specified data channel,
            primarily relevant for non-multiplex measurements
        """
        return self.data.iloc[dataChannelIndex]['channelName']

    def get_data_channel_index(self, dataChannelName: str) -> int:
        """Get the index for the data channel with the specified name.

        Args:
            dataChannelName: the name of the data channel. The data channel
                name is not case sensitive.
        Returns:
            the index of the data channel where the data channel name is
                dataChannelName
        Raises:
            # TODO this should raise a meaningful exception if the data channel
            # is not found
        """
        return self.data[self.data['channelName'].apply(
            lambda x: str(x).lower()) == str(dataChannelName).lower()]\
            .index.values.tolist()[0]

    def get_data_channel_color(self, dataChannel: int) -> str:
        """Get the color used for measuring the specified data channel.

        Args:
            dataChannel: the data channel index
        Returns:
            the color for this data channel as a string
        """
        return str(self.data.at[dataChannel, 'color'])

    def get_data_channel_for_bit(self, bitName: str) -> int:
        """Get the data channel associated with the specified bit.

        Args:
            bitName: the name of the bit to search for
        Returns:
            The index of the associated data channel
        """
        return self.data[self.data['readoutName'] ==
                         bitName].index.values.item()

    def get_data_channel_with_name(self, channelName: str) -> int:
        """Get the data channel associated with a gene name.

        Args:
            channelName: the name of the gene to search for
        Returns:
            The index of the associated data channel
        """
        return self.data[self.data['channelName'] ==
                         channelName].index.values.item()

    def get_fiducial_filename(self, dataChannel: int, fov: int) -> str:
        """Get the path for the image file that contains the fiducial
        image for the specified dataChannel and fov.

        Args:
            dataChannel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """

        imageType = self.data.loc[dataChannel, 'fiducialImageType']
        imagingRound = \
            self.data.loc[dataChannel, 'fiducialImagingRound']
        return self._get_image_path(imageType, fov, imagingRound)

    def get_fiducial_frame_index(self, dataChannel: int) -> int:
        """Get the index of the frame containing the fiducial image
        for the specified data channel.

        Args:
            dataChannel: index of the data channel
        Returns:
            The index of the fiducial frame in the corresponding image file
        """
        return self.data.iloc[dataChannel]['fiducialFrame']

    def get_image_filename(self, dataChannel: int, fov: int) -> str:
        """Get the path for the image file that contains the
        images for the specified dataChannel and fov.

        Args:
            dataChannel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """
        channelInfo = self.data.iloc[dataChannel]
        imagePath = self._get_image_path(
                channelInfo['imageType'], fov, channelInfo['imagingRound'])
        return imagePath

    def get_image_frame_index(self, dataChannel: int, zPosition: float) -> int:
        """Get the index of the frame containing the image
        for the specified data channel and z position.

        Args:
            dataChannel: index of the data channel
            zPosition: the z position
        Returns:
            The index of the frame in the corresponding image file
        """
        channelInfo = self.data.iloc[dataChannel]
        channelZ = channelInfo['zPos']
        if isinstance(channelZ, np.ndarray):
            zIndex = np.where(channelZ == zPosition)[0]
            if len(zIndex) == 0:
                raise Exception('Requested z position not found. Position ' +
                                'z=%0.2f not found for channel %i'
                                % (zPosition, dataChannel))
            else:
                frameIndex = zIndex[0]
        else:
            frameIndex = 0

        frames = channelInfo['frame']
        if isinstance(frames, np.ndarray):
            frame = frames[frameIndex]
        else:
            frame = frames

        return frame

    def get_z_positions(self) -> List[float]:
        """Get the z positions present in this data organization.

        Returns:
            A sorted list of all unique z positions
        """
        return sorted(np.unique([y for x in self.data['zPos'] for y in x]))

    def get_fovs(self) -> np.ndarray:
        fovs = np.unique(self.fileMap['fov'])
        if self._max_fov is not None:
            fovs = fovs[fovs < self._max_fov]
        return fovs

    # only used for tiff directory, not zarr
    def _get_image_path(
            self, imageType: str, fov: int, imagingRound: int) -> str:
        selection = self.fileMap[(self.fileMap['imageType'] == imageType) &
                                 (self.fileMap['fov'] == fov) &
                                 (self.fileMap['imagingRound'] == imagingRound)]
        filemapPath = selection['imagePath'].values[0]
        return os.path.join(self._data_path, filemapPath)

    # def _truncate_file_path(self, path) -> None:
    #     head, tail = os.path.split(path)
    #     return tail

    # no longer userd
    # def _map_image_files(self, extension_list=['.tif','.tiff','.zarr']) -> None:
    #     # TODO: This doesn't map the fiducial image types and currently assumes
    #     # that the fiducial image types and regular expressions are part of the
    #     # standard image types.

    #     #try:
    #     #    self.fileMap = self._dataSet.load_dataframe_from_csv('filemap')
    #     #    self.fileMap['imagePath'] = self.fileMap['imagePath'].apply(
    #     #        self._truncate_file_path)

    #     #except FileNotFoundError:
    #     uniqueEntries = self.data.drop_duplicates(
    #         subset=['imageType', 'imageRegExp'], keep='first')

    #     uniqueTypes = uniqueEntries['imageType']
    #     uniqueIndexes = uniqueEntries.index.values.tolist()

    #     # list the files in self._data_path with any of the extensions in extension_list
    #     fileNames = []
    #     for file in os.listdir(self._data_path):
    #         if any([file.endswith(ext) for ext in extension_list]):
    #             fileNames.append(os.path.join(self._data_path, file))

    #     if len(fileNames) == 0:
    #         raise DataFormatException(
    #             'No image files found at %s.' % self._data_path)
    #     fileData = []
    #     for currentType, currentIndex in zip(uniqueTypes, uniqueIndexes):
    #         matchRE = re.compile(
    #                 self.data.imageRegExp[currentIndex])

    #         matchingFiles = False
    #         for currentFile in fileNames:
    #             matchedName = matchRE.match(os.path.split(currentFile)[-1])
    #             if matchedName is not None:
    #                 transformedName = matchedName.groupdict()
    #                 if transformedName['imageType'] == currentType:
    #                     if 'imagingRound' not in transformedName:
    #                         transformedName['imagingRound'] = -1
    #                     transformedName['imagePath'] = currentFile
    #                     matchingFiles = True
    #                     fileData.append(transformedName)

    #         if not matchingFiles:
    #             raise DataFormatException(
    #                 'Unable to identify image files matching regular '
    #                 + 'expression %s for image type %s.'
    #                 % (self.data.imageRegExp[currentIndex],
    #                     currentType))

    #     self.fileMap = pandas.DataFrame(fileData)
    #     self.fileMap[['imagingRound', 'fov']] = \
    #         self.fileMap[['imagingRound', 'fov']].astype(int)
    #     self.fileMap['imagePath'] = self.fileMap['imagePath'].apply(
    #         self._truncate_file_path)

    #     self._validate_file_map()

    #     #self._dataSet.save_dataframe_to_csv(
    #     #        self.fileMap, 'filemap', index=False)

    # def _validate_file_map(self) -> None:
    #     """
    #     This function ensures that all the files specified in the file map
    #     of the raw images are present.

    #     Raises:
    #         InputDataError: If the set of raw data is incomplete or the
    #                 format of the raw data deviates from expectations.
    #     """

    #     expectedImageSize = None
    #     for dataChannel in self.get_data_channels():
    #         for fov in tqdm(self.get_fovs()):
    #             #if self._max_fov is not None and fov < self._max_fov:
    #             channelInfo = self.data.iloc[dataChannel]
    #             try:
    #                 imagePath = self._get_image_path(
    #                     channelInfo['imageType'], fov,
    #                     channelInfo['imagingRound'])
    #             except IndexError:
    #                 raise FileNotFoundError(
    #                     'Unable to find image path for %s, fov=%i, round=%i' %
    #                     (channelInfo['imageType'], fov,
    #                     channelInfo['imagingRound']))

    #             if not os.path.exists(imagePath):
    #                 raise InputDataError(
    #                     ('Image data for channel {0} and fov {1} not found. '
    #                     'Expected at {2}')
    #                     .format(dataChannel, fov, imagePath))
    #         # check the size of just the first image file from the first FOV -- assume that others are correct
    #     dataChannel = self.get_data_channels()[0]
    #     fov = self.get_fovs()[0]
    #     try:
    #         imageSize = imread(imagePath).shape
    #     except Exception as e:
    #         raise InputDataError(
    #             ('Unable to determine image stack size for fov {0} from'
    #                 ' data channel {1} at {2}')
    #             .format(dataChannel, fov, imagePath))

    #     frames = channelInfo['frame']

    #     # this assumes fiducials are stored in the same image file
    #     requiredFrames = max(np.max(frames),
    #                             channelInfo['fiducialFrame'])
    #     if requiredFrames >= imageSize[2]:
    #         raise InputDataError(
    #             ('Insufficient frames in data for channel {0} and '
    #                 'fov {1}. Expected {2} frames '
    #                 'but only found {3} in file {4}')
    #             .format(dataChannel, fov, requiredFrames, imageSize[2],
    #                     imagePath))

    #     if expectedImageSize is None:
    #         expectedImageSize = [imageSize[0], imageSize[1]]
    #     else:
    #         if expectedImageSize[0] != imageSize[0] \
    #                 or expectedImageSize[1] != imageSize[1]:
    #             raise InputDataError(
    #                 ('Image data for channel {0} and fov {1} has '
    #                     'unexpected dimensions. Expected {1}x{2} but '
    #                     'found {3}x{4} in image file {5}')
    #                 .format(dataChannel, fov, expectedImageSize[0],
    #                         expectedImageSize[1], imageSize[0],
    #                         imageSize[1], imagePath))