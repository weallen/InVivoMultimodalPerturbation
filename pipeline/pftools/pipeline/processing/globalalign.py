import numpy as np
from typing import Tuple, List, Optional, Union
from shapely import geometry

from pftools.pipeline.core.algorithm import AnalysisTask
import os
from abc import abstractmethod
import numpy as np
from typing import Tuple
from typing import List
from shapely.geometry import box

class SimpleGlobalAlignment:

    """A global alignment that uses the theoretical stage positions in
    order to determine the relative positions of each field of view.
    """

    def __init__(self, output_path:str, fov_offsets:np.ndarray, 
                 z_pos:np.ndarray,
                 micron_per_pixel:float=0.109, 
                 image_dimensions:Tuple[int, int]=(2048, 2048)):
        """
        fov_offsets is list of expected stage positions in the global coordinated system 
        """
        self.output_path = os.path.join(output_path, self.__class__.__name__)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.fovs = np.arange(len(fov_offsets)).astype(int)
        self.fov_offsets = [list(i) for i in fov_offsets]
        self.microns_per_pixel = micron_per_pixel
        self.image_dimensions = image_dimensions
        self.z_pos = z_pos

    def fov_coordinates_to_global(self, fov:int, fovCoordinates: Tuple[np.ndarray, np.ndarray]) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Calculates the global coordinates based on the local coordinates
        in the specified field of view.

        Args:
            fov: the fov where the coordinates are measured
            fovCoordinates: a tuple containing the x and y coordinates
                or z, x, and y coordinates (in pixels) in the specified fov.
        Returns:
            A tuple containing the global x and y coordinates or
            z, x, and y coordinates (in microns)
        """

        fovStart = self.fov_offsets[fov]
        micronsPerPixel = self.microns_per_pixel
        if len(fovCoordinates) == 2:
            return (fovStart[0] + fovCoordinates[0]*micronsPerPixel,
                    fovStart[1] + fovCoordinates[1]*micronsPerPixel)
        elif len(fovCoordinates) == 3:
            zPositions = self.z_pos
            return (np.interp(fovCoordinates[0], np.arange(len(zPositions)),
                              zPositions),
                    fovStart[0] + fovCoordinates[1]*micronsPerPixel,
                    fovStart[1] + fovCoordinates[2]*micronsPerPixel)
        else:
            return (np.zeros(1), np.zeros(1))

    def fov_coordinate_array_to_global(self, fov: int,
                                       fovCoordArray: np.ndarray) -> np.ndarray:
        """A bulk transformation of a list of fov coordinates to
           global coordinates.
        Args:
            fov: the fov of interest
            fovCoordArray: numpy array of the [z, x, y] positions to transform
        Returns:
            numpy array of the global [z, x, y] coordinates
        """

        tForm = self.fov_to_global_transform(fov)
        toGlobal = np.ones(fovCoordArray.shape)
        toGlobal[:, [0, 1]] = fovCoordArray[:, [1, 2]]
        globalCentroids = np.matmul(tForm, toGlobal.T).T[:, [2, 0, 1]]
        globalCentroids[:, 0] = fovCoordArray[:, 0]
        return globalCentroids

    def fov_global_extent(self, fov: int) -> List[float]:
        """
        Returns the global extent of a fov, output interleaved as
        xmin, ymin, xmax, ymax

        Args:
            fov: the fov of interest
        Returns:
            a list of four floats, representing the xmin, xmax, ymin, ymax
        """

        return [x for y in (self.fov_coordinates_to_global(fov, (0, 0)),
                            self.fov_coordinates_to_global(fov, (self.image_dimensions[0], self.image_dimensions[1])))
                for x in y] # type: ignore

    def global_coordinates_to_fov(self, fov, globalCoordinates):
        tform = np.linalg.inv(self.fov_to_global_transform(fov))

        def convert_coordinate(coordinateIn):
            coords = np.array([coordinateIn[0], coordinateIn[1], 1])
            return np.matmul(tform, coords).astype(int)[:2]
        pixels = [convert_coordinate(x) for x in globalCoordinates]
        return pixels

    def fov_to_global_transform(self, fov):
        """Calculates the transformation matrix for an affine transformation
        that transforms the fov coordinates to global coordinates.

        Args:
            fov: the fov to calculate the transformation
        Returns:
            a numpy array containing the transformation matrix
        """

        micronsPerPixel = self.microns_per_pixel
        globalStart = self.fov_coordinates_to_global(fov, (0, 0))

        return np.float32([[micronsPerPixel, 0, globalStart[0]],
                           [0, micronsPerPixel, globalStart[1]],
                           [0, 0, 1]]) # type: ignore

    def get_global_extent(self):
        fovSize = self.image_dimensions
        fovBounds = [self.fov_coordinates_to_global(x, (0, 0))
                     for x in self.fovs] + \
                    [self.fov_coordinates_to_global(x, fovSize)
                     for x in self.fovs]

        minX = np.min([x[0] for x in fovBounds]) # type: ignore
        maxX = np.max([x[0] for x in fovBounds]) # type: ignore
        minY = np.min([x[1] for x in fovBounds])
        maxY = np.max([x[1] for x in fovBounds])

        return minX, minY, maxX, maxY


    def get_fov_boxes(self) -> List:
        """
        Creates a list of shapely boxes for each fov containing the global
        coordinates as the box coordinates.

        Returns:
            A list of shapely boxes
        """
        boxes = [geometry.box(*self.fov_global_extent(f)) for f in self.fovs] # type: ignore

        return boxes
