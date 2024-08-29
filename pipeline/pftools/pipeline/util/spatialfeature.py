from abc import abstractmethod
import numpy as np
import uuid
import cv2
from skimage import measure
from typing import List, Tuple, Dict, Optional
from shapely import geometry
import h5py
import pandas
import networkx as nx
import rtree
from scipy.spatial import cKDTree
import geopandas as geo 

#def load_spatial_features_from_parquet(path: str) -> List['SpatialFeature']:
#    df = geo.read_parquet(path)

class SpatialFeature(object):

    """
    A spatial feature is a collection of contiguous voxels.
    """

    def __init__(self, boundaryList: List[List[geometry.Polygon]], fov: int,
                 zCoordinates: Optional[np.ndarray] = None, uniqueID: Optional[int] = None,
                 label: int = -1) -> None:
        """Create a new feature specified by a list of pixels

        Args:
            boundaryList: a list of boundaries that define this feature.
                The first index of the list corresponds with the z index.
                The second index corresponds with the index of the shape since
                some regions might split in some z indexes.
            fov: the index of the field of view that this feature belongs to.
                The pixel list specifies pixel in the local fov reference
                frame.
            zCoordinates: the z position for each of the z indexes. If not
                specified, each z index is assumed to have unit height.
            uniqueID: the uuid of this feature. If no uuid is specified,
                a new uuid is randomly generated.
            label: unused
        """
        self._boundaryList = boundaryList
        self._fov = fov

        if uniqueID is None:
            self._uniqueID = uuid.uuid4().int
        else:
            self._uniqueID = uniqueID

        if zCoordinates is not None:
            self._zCoordinates = zCoordinates
        else:
            self._zCoordinates = np.arange(len(boundaryList))

    @staticmethod
    def feature_from_label_matrix(labelMatrix: np.ndarray, fov: int,
                                  transformationMatrix: Optional[np.ndarray] = None,
                                  zCoordinates: Optional[np.ndarray] = None,
                                  label: int = -1) -> 'SpatialFeature':
        """Generate a new feature from the specified label matrix.

        Args:
            labelMatrix: a 3d matrix indicating the z, x, y position
                of voxels that contain the feature. Voxels corresponding
                to the feature have a value of True while voxels outside of the
                feature should have a value of False.
            fov: the index of the field of view corresponding to the
                label matrix.
            transformationMatrix: a 3x3 numpy array specifying the
                transformation from fov to global coordinates. If None,
                the feature coordinates are not transformed.
            zCoordinates: the z position for each of the z indexes. If not
                specified, each z index is assumed to have unit height.
        Returns: the new feature
        """

        boundaries = [SpatialFeature._extract_boundaries(x)
                      for x in labelMatrix]

        if transformationMatrix is not None:
            boundaries = [SpatialFeature._transform_boundaries(
                x, transformationMatrix) for x in boundaries]

        return SpatialFeature([SpatialFeature._remove_invalid_boundaries(
            SpatialFeature._remove_interior_boundaries(
                [geometry.Polygon(x) for x in b if len(x) > 2]))
                               for b in boundaries], fov, zCoordinates)

    @staticmethod
    def _extract_boundaries(labelMatrix: np.ndarray) -> List[np.ndarray]:
        """Determine the boundaries of the feature indicated in the
        label matrix.

        Args:
            labelMatrix: a 2 dimensional numpy array indicating the x, y
                position of pixels that contain the feature.
        Returns: a list of n x 2 numpy arrays indicating the x, y coordinates
            of the boundaries where n is the number of boundary coordinates
        """
        boundaries = measure.find_contours(np.transpose(labelMatrix), 0.9,
                                           fully_connected='high')
        return boundaries

    @staticmethod
    def _transform_boundaries(
            boundaries: List[np.ndarray],
            transformationMatrix: np.ndarray) -> List[np.ndarray]:

        transformedList = []
        for b in boundaries:
            reshapedBoundaries = np.reshape(
                b, (1, b.shape[0], 2)).astype(np.float32)
            transformedBoundaries = cv2.transform(
                reshapedBoundaries, transformationMatrix)[0, :, :2]
            transformedList.append(transformedBoundaries)

        return transformedList

    @staticmethod
    def _remove_interior_boundaries(
            inPolygons: List[geometry.Polygon]) -> List[geometry.Polygon]:
        goodPolygons = []

        for p in inPolygons:
            if not any([pTest.contains(p)
                        for pTest in inPolygons if p != pTest]):
                goodPolygons.append(p)

        return goodPolygons

    @staticmethod
    def _remove_invalid_boundaries(
            inPolygons: List[geometry.Polygon]) -> List[geometry.Polygon]:
        return [p for p in inPolygons if p.is_valid]

    def set_fov(self, newFOV: int) -> None:
        """Update the FOV for this spatial feature.

        Args:
            nowFOV: the new FOV index
        """
        self._fov = newFOV

    def get_fov(self) -> int:
        return self._fov

    def get_boundaries(self) -> List[List[geometry.Polygon]]:
        return self._boundaryList

    def get_feature_id(self) -> int:
        return self._uniqueID

    def get_z_coordinates(self) -> np.ndarray:
        return self._zCoordinates

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the 2d box that contains all boundaries in all z plans of this
        feature.

        Returns:
            a tuple containing (x1, y1, x2, y2) coordinates of the bounding box
        """
        boundarySet = []
        for f in self.get_boundaries():
            for b in f:
                boundarySet.append(b)

        multiPolygon = geometry.MultiPolygon(boundarySet)
        return multiPolygon.bounds

    def get_volume(self) -> float:
        """Get the volume enclosed by this feature.

        Returns:
            the volume represented in global coordinates. If only one z
            slice is present for the feature, the z height is taken as 1.
        """
        boundaries = self.get_boundaries()

        zPos = np.array(self._zCoordinates)
        if len(zPos) > 1:
            zDiff = np.diff(zPos)
            zNum = np.array([[x, x + 1] for x in range(len(zPos) - 1)])
            areas = np.array([np.sum([y.area for y in x]) if len(x) > 0
                              else 0 for x in boundaries])
            totalVolume = np.sum([np.mean(areas[zNum[x]]) * zDiff[x]
                                  for x in range(zNum.shape[0])])
        else:
            totalVolume = np.sum([y.area for x in boundaries for y in x])

        return totalVolume

    def intersection(self, intersectFeature) -> float:

        intersectArea = 0
        for p1Set, p2Set in zip(self.get_boundaries(),
                                intersectFeature.get_boundaries()):
            for p1 in p1Set:
                for p2 in p2Set:
                    intersectArea += p1.intersection(p2).area

        return intersectArea

    def is_contained_within_boundary(self, inFeature) -> bool:
        """Determine if any part of this feature is contained within the
        boundary of the specified feature.

        Args:
            inFeature: the feature whose boundary should be checked whether
                it contains this feature
        Returns:
            True if inFeature contains pixels that are within inFeature,
                otherwise False. This returns false if inFeature only shares
                a boundary with this feature.
        """
        if all([b1.disjoint(b2) for b1List, b2List in zip(
                    self.get_boundaries(), inFeature.get_boundaries())
                for b1 in b1List for b2 in b2List]):
            return False

        for b1List, b2List in zip(
                self.get_boundaries(), inFeature.get_boundaries()):
            for b1 in b1List:
                for b2 in b2List:
                    x, y = b1.exterior.coords.xy
                    for p in zip(x, y):
                        if geometry.Point(p).within(b2):
                            return True

        return False

    def equals(self, testFeature) -> bool:
        """Determine if this feature is equivalent to testFeature

        Args:
            testFeature: the feature to test equivalency
        Returns:
            True if this feature and testFeature are equivalent, otherwise
                false
        """
        if self.get_fov() != testFeature.get_fov():
            return False
        if self.get_feature_id() != testFeature.get_feature_id():
            return False
        if not np.array_equal(self.get_z_coordinates(),
                              testFeature.get_z_coordinates()):
            return False

        if len(self.get_boundaries()) != len(testFeature.get_boundaries()):
            return False
        for b, bIn in zip(self.get_boundaries(), testFeature.get_boundaries()):
            if len(b) != len(bIn):
                return False
            for x, y in zip(b, bIn):
                if not x.equals(y):
                    return False

        return True

    def contains_point(self, point: geometry.Point, zIndex: int) -> bool:
        """Determine if this spatial feature contains the specified point.

        Args:
            point: the point to check
            zIndex: the z-index that the point corresponds to
        Returns:
            True if the boundaries of this spatial feature in the zIndex plane
                contain the given point.
        """
        for boundaryElement in self.get_boundaries()[zIndex]:
            if boundaryElement.contains(point):
                return True

        return False

    def contains_positions(self, positionList: np.ndarray) -> np.ndarray:
        """Determine if this spatial feature contains the specified positions

        Args:
            positionList: a N x 3 numpy array containing the (x, y, z)
                positions for N points where x and y are spatial coordinates
                and z is the z index. If z is not an integer it is rounded
                to the nearest integer.
        Returns:
            a numpy array of booleans containing true in the i'th index if
                the i'th point provided is in this spatial feature.
        """
        boundaries = self.get_boundaries()
        positionList[:, 2] = np.round(positionList[:, 2])

        containmentList = np.zeros(positionList.shape[0], dtype=np.bool8)

        for zIndex in range(len(boundaries)):
            currentIndexes = np.where(positionList[:, 2] == zIndex)[0]
            currentContainment = [self.contains_point(
                geometry.Point(x[0], x[1]), zIndex)
                for x in positionList[currentIndexes]]
            containmentList[currentIndexes] = currentContainment

        return containmentList

    def get_overlapping_features(self, featuresToCheck: List['SpatialFeature']
                                 ) -> List['SpatialFeature']:
        """ Determine which features within the provided list overlap with this
        feature.

        Args:
            featuresToCheck: the list of features to check for overlap with
                this feature.
        Returns: the features that overlap with this feature
        """
        areas = [self.intersection(x) for x in featuresToCheck]
        overlapping = [featuresToCheck[i] for i, x in enumerate(areas) if x > 0]
        benchmark = self.intersection(self)
        contained = [x for x in overlapping if
                     x.intersection(self) == benchmark]
        if len(contained) > 1:
            overlapping = []
        else:
            toReturn = []
            for c in overlapping:
                if c.get_feature_id() == self.get_feature_id():
                    toReturn.append(c)
                else:
                    if c.intersection(self) != c.intersection(c):
                        toReturn.append(c)
            overlapping = toReturn

        return overlapping

    def to_geopandas(self) -> geo.GeoDataFrame:
        z_planes = []
        z_idx = self.get_z_coordinates()
        for z, boundaries in enumerate(self.get_boundaries()):
            for boundary in boundaries:
                z_planes.append((z_idx[z], boundary))
        x = [float(np.array(b[1].centroid.xy[0])) for b in z_planes]
        y = [float(np.array(b[1].centroid.xy[1])) for b in z_planes]
        n_boundaries = len(z_planes)
        geodf = geo.GeoDataFrame({
            'fov': [self.get_fov()]*n_boundaries,
            'x': x,
            'y' : y,
            'id': [self.get_feature_id()]*n_boundaries,
            'z' : [z[0] for z in z_planes],
        }, geometry=[z[1] for z in z_planes])
        return geodf

    @staticmethod
    def from_geopandas(geodf: geo.GeoDataFrame) -> 'SpatialFeature':
        raise NotImplementedError()

    @staticmethod 
    def from_geopandas(self, gdf: geo.GeoDataFrame) -> 'SpatialFeature':
        pass
        #sf = SpatialFeature([], 0, np.array([]), 
        #return sf

    def to_json_dict(self) -> Dict:
        return {
            'fov': self._fov,
            'id': self._uniqueID,
            'z_coordinates': self._zCoordinates.tolist(),
            'boundaries': [[geometry.mapping(y) for y in x]
                           for x in self.get_boundaries()]
        }

    @staticmethod
    def from_json_dict(jsonIn: Dict):
        boundaries = [[geometry.shape(y) for y in x]
                      for x in jsonIn['boundaries']]

        return SpatialFeature(boundaries,
                              jsonIn['fov'],
                              np.array(jsonIn['z_coordinates']),
                              jsonIn['id'])


def simple_clean_cells(cells: List[SpatialFeature]) -> List[SpatialFeature]:
    """
    Removes cells that lack a bounding box or have a volume equal to 0

    Args:
        cells: List of spatial features

    Returns:
        List of spatial features

    """
    return [cell for cell in cells
            if len(cell.get_bounding_box()) == 4 and cell.get_volume() > 0]


def append_cells_to_spatial_tree(tree: rtree.index.Index,
                                 cells: List, idToNum: Dict):
    for element in cells:
        tree.insert(idToNum[element.get_feature_id()],
                    element.get_bounding_box(), obj=element)


def construct_tree(cells: List,
                   spatialIndex: rtree.index.Index = rtree.index.Index(),
                   count: int = 0, idToNum: Dict = dict()):
    """
    Builds or adds to an rtree with a list of cells

    Args:
        cells: list of spatial features
        spatialIndex: an existing rtree to append to
        count: number of existing entries in existing rtree
        idToNum: dict containing feature ID as key, and number in rtree as value

    Returns:
        spatialIndex: an rtree updated with the input cells
        count: number of entries in rtree
        idToNum: dict containing feature ID as key, and number in rtree as value
    """

    for i in range(len(cells)):
        idToNum[cells[i].get_feature_id()] = count
        count += 1
    append_cells_to_spatial_tree(spatialIndex, cells, idToNum)

    return spatialIndex, count, idToNum


def return_overlapping_cells(currentCell, cells: List):
    """
    Determines if there is overlap between a cell of interest and a list of
    other cells. In the event that the cell of interest is entirely contained
    within one of the cells in the cells it is being compared to, an empty
    list is returned. Otherwise, the cell of interest and any overlapping
    cells are returned.
    Args:
        currentCell: A spatial feature of interest
        cells: A list of spatial features to compare to, the spatial feature
               of interest is expected to be in this list

    Returns:
        A list of spatial features including the cell of interest and all
        overlapping cells, or an empty list if the cell of intereset is
        entirely contained within one of the cells it is compared to
    """
    areas = [currentCell.intersection(x) for x in cells]
    overlapping = [cells[i] for i, x in enumerate(areas) if x > 0]
    benchmark = currentCell.intersection(currentCell)
    contained = [x for x in overlapping if
                 x.intersection(currentCell) == benchmark]
    if len(contained) > 1:
        overlapping = []
    else:
        toReturn = []
        for c in overlapping:
            if c.get_feature_id() == currentCell.get_feature_id():
                toReturn.append(c)
            else:
                if c.intersection(currentCell) != c.intersection(c):
                    toReturn.append(c)
        overlapping = toReturn

    return overlapping


def construct_graph(graph, cells, spatialTree, currentFOV, allFOVs, fovBoxes):
    """
    Adds the cells from the current fov to a graph where each node is a cell
    and edges connect overlapping cells.

    Args:
        graph: An undirected graph, either empty of already containing cells
        cells: A list of spatial features to potentially add to graph
        spatialTree: an rtree index containing each cell in the dataset
        currentFOV: the fov currently being added to the graph
        allFOVs: a list of all fovs in the dataset
        fovBoxes: a list of shapely polygons containing the bounds of each fov

    Returns:
        A graph updated to include cells from the current fov
    """

    fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                               fovBoxes[currentFOV].intersects(x)])

    coords = [x.centroid.coords.xy for x in fovBoxes]
    xcoords = [x[0][0] for x in coords]
    ycoords = [x[1][0] for x in coords]
    coordsDF = pandas.DataFrame(data=np.array(list(zip(xcoords, ycoords))),
                                index=allFOVs,
                                columns=['centerX', 'centerY'])
    fovTree = cKDTree(data=coordsDF.loc[fovIntersections,
                                        ['centerX', 'centerY']].values)
    for cell in cells:
        overlappingCells = spatialTree.intersection(
            cell.get_bounding_box(), objects=True)
        toCheck = [x.object for x in overlappingCells]
        cellsToConsider = return_overlapping_cells(
            cell, toCheck)
        if len(cellsToConsider) == 0:
            pass
        else:
            for cellToConsider in cellsToConsider:
                xmin, ymin, xmax, ymax =\
                    cellToConsider.get_bounding_box()
                xCenter = (xmin + xmax) / 2
                yCenter = (ymin + ymax) / 2
                [d, i] = fovTree.query(np.array([xCenter, yCenter]))
                assignedFOV = coordsDF.loc[fovIntersections, :]\
                    .index.values.tolist()[i]
                if cellToConsider.get_feature_id() not in graph.nodes:
                    graph.add_node(cellToConsider.get_feature_id(),
                                   originalFOV=cellToConsider.get_fov(),
                                   assignedFOV=assignedFOV)
            if len(cellsToConsider) > 1:
                for cellToConsider1 in cellsToConsider:
                    if cellToConsider1.get_feature_id() !=\
                            cell.get_feature_id():
                        graph.add_edge(cell.get_feature_id(),
                                       cellToConsider1.get_feature_id())
    return graph


def remove_overlapping_cells(graph):
    """
    Takes in a graph in which each node is a cell and edges connect cells that
    overlap eachother in space. Removes overlapping cells, preferentially
    eliminating the cell that overlaps the most cells (i.e. if cell A overlaps
    cells B, C, and D, whereas cell B only overlaps cell A, cell C only overlaps
    cell A, and cell D only overlaps cell A, then cell A will be removed,
    leaving cells B, C, and D remaining because there is no more overlap
    within this group of cells).
    Args:
        graph: An undirected graph, in which each node is a cell and each
               edge connects overlapping cells. nodes are expected to have
               the following attributes: originalFOV, assignedFOV
    Returns:
        A pandas dataframe containing the feature ID of all cells after removing
        all instances of overlap. There are columns for cell_id, originalFOV,
        and assignedFOV
    """
    connectedComponents = list(nx.connected_components(graph))
    cleanedCells = []
    connectedComponents = [list(x) for x in connectedComponents]
    for component in connectedComponents:
        if len(component) == 1:
            originalFOV = graph.nodes[component[0]]['originalFOV']
            assignedFOV = graph.nodes[component[0]]['assignedFOV']
            cleanedCells.append([component[0], originalFOV, assignedFOV])
        if len(component) > 1:
            sg = nx.subgraph(graph, component)
            verts = list(nx.articulation_points(sg))
            if len(verts) > 0:
                sg = nx.subgraph(graph,
                                 [x for x in component if x not in verts])
            allEdges = [[k, v] for k, v in nx.degree(sg)]
            sortedEdges = sorted(allEdges, key=lambda x: x[1], reverse=True)
            maxEdges = sortedEdges[0][1]
            while maxEdges > 0:
                sg = nx.subgraph(graph, [x[0] for x in sortedEdges[1:]])
                allEdges = [[k, v] for k, v in nx.degree(sg)]
                sortedEdges = sorted(allEdges, key=lambda x: x[1],
                                     reverse=True)
                maxEdges = sortedEdges[0][1]
            keptComponents = list(sg.nodes())
            cellIDs = []
            originalFOVs = []
            assignedFOVs = []
            for c in keptComponents:
                cellIDs.append(c)
                originalFOVs.append(graph.nodes[c]['originalFOV'])
                assignedFOVs.append(graph.nodes[c]['assignedFOV'])
            listOfLists = list(zip(cellIDs, originalFOVs, assignedFOVs))
            listOfLists = [list(x) for x in listOfLists]
            cleanedCells = cleanedCells + listOfLists
    cleanedCellsDF = pandas.DataFrame(cleanedCells,
                                      columns=['cell_id', 'originalFOV',
                                               'assignedFOV'])
    return cleanedCellsDF