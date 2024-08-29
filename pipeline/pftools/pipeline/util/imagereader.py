import hashlib
import numpy as np
import re
import tifffile
from typing import List
import dask.array as da
import skimage

from pftools.pipeline.util import dataportal


def load_dask_stack_from_filenames(filenames, shape, dtype=np.uint16):
    """
    Load stacked dask array from list of filenames
    Need to know shape ahead of time
    """
    if "tif" in filenames[0] or "tiff" in filenames[0]:
        def read_one_image(block_id, filenames=filenames, axis=0):
            # a function that reads in one chunk of data
            path = filenames[block_id[axis]]
            image = skimage.io.imread(path)
            return np.expand_dims(image, axis=axis)
        
        stack = da.map_blocks(
            read_one_image,
            dtype=dtype,
            chunks=((1,) * len(filenames),  *shape)
        )
        return stack
    else:
        raise NotImplementedError("Only TIFF files are supported right now")


def infer_reader(filePortal: dataportal.FilePortal, verbose: bool = False):
    """
    Given a file name this will try to return the appropriate
    reader based on the file extension.
    """
    ext = filePortal.get_file_extension()

    if ext == '.dax':
        return DaxReader(filePortal, verbose=verbose)
    elif ext == ".zarr":
        return ZarrReader(filePortal._fileName, verbose=verbose)
    elif ext == ".tif" or ext == ".tiff":
        return TifReader(filePortal._fileName, verbose=verbose)
    
    raise IOError(
        "only .dax, .zarr, and .tif are supported (case sensitive..)")


class Reader(object):
    """
    The superclass containing those functions that
    are common to reading a STORM movie file.
    Subclasses should implement:
     1. __init__(self, filename, verbose = False)
        This function should open the file and extract the
        various key bits of meta-data such as the size in XY
        and the length of the movie.
     2. loadAFrame(self, frame_number)
        Load the requested frame and return it as np array.
    """

    def __init__(self, filename, verbose=False):
        super(Reader, self).__init__()
        self.image_height = 0
        self.image_width = 0
        self.number_frames = 0
        self.stage_x = 0
        self.stage_y = 0
        self.filename = filename
        self.fileptr = None
        self.verbose = verbose

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.close()

    def average_frames(self, start=None, end=None):
        """
        Average multiple frames in a movie.
        """
        length = 0
        average = np.zeros((self.image_height, self.image_width),
                           np.float)
        for [i, frame] in self.frame_iterator(start, end):
            if self.verbose and ((i % 10) == 0):
                print(" processing frame:", i, " of", self.number_frames)
            length += 1
            average += frame

        if length > 0:
            average = average / float(length)

        return average

    def close(self):
        if self.fileptr is not None:
            self.fileptr.close()
            self.fileptr = None

    def film_filename(self):
        """
        Returns the film name.
        """
        return self.filename

    def film_size(self):
        """
        Returns the film size.
        """
        return [self.image_width, self.image_height, self.number_frames]

    def film_location(self):
        """
        Returns the picture x,y location, if available.
        """
        if hasattr(self, "stage_x"):
            return [self.stage_x, self.stage_y]
        else:
            return [0.0, 0.0]

    def film_scale(self):
        """
        Returns the scale used to display the film when
        the picture was taken.
        """
        if hasattr(self, "scalemin") and hasattr(self, "scalemax"):
            return [self.scalemin, self.scalemax]
        else:
            return [100, 2000]

    def frame_iterator(self, start=None, end=None):
        """
        Iterator for going through the frames of a movie.
        """
        if start is None:
            start = 0
        if end is None:
            end = self.number_frames

        for i in range(start, end):
            yield [i, self.load_frame(i)]

    def hash_ID(self):
        """
        A (hopefully) unique string that identifies this movie.
        """
        return hashlib.md5(self.load_frame(0).tostring()).hexdigest()

    def load_frame(self, frame_number):
        assert frame_number >= 0, \
            "Frame_number must be greater than or equal to 0, it is "\
            + str(frame_number)
        assert frame_number < self.number_frames, \
            "Frame number must be less than " + str(self.number_frames)

    def lock_target(self):
        """
        Returns the film focus lock target.
        """
        if hasattr(self, "lock_target"):
            return self.lock_target
        else:
            return 0.0


class DaxReader(Reader):
    """
    Dax reader class. This is a Zhuang lab custom format.
    """

    def __init__(self, filePortal: dataportal.FilePortal,
                 verbose: bool = False):
        super(DaxReader, self).__init__(
            filePortal.get_file_name(), verbose=verbose)

        self._filePortal = filePortal
        infFile = filePortal.get_sibling_with_extension('.inf')
        self._parse_inf(infFile.read_as_text().splitlines())

    def close(self):
        self._filePortal.close()

    def _parse_inf(self, inf_lines: List[str]) -> None:
        size_re = re.compile(r'frame dimensions = ([\d]+) x ([\d]+)')
        length_re = re.compile(r'number of frames = ([\d]+)')
        endian_re = re.compile(r' (big|little) endian')
        stagex_re = re.compile(r'Stage X = ([\d.\-]+)')
        stagey_re = re.compile(r'Stage Y = ([\d.\-]+)')
        lock_target_re = re.compile(r'Lock Target = ([\d.\-]+)')
        scalemax_re = re.compile(r'scalemax = ([\d.\-]+)')
        scalemin_re = re.compile(r'scalemin = ([\d.\-]+)')

        # defaults
        self.image_height = None
        self.image_width = None

        for line in inf_lines:
            m = size_re.match(line)
            if m:
                self.image_height = int(m.group(2))
                self.image_width = int(m.group(1))
            m = length_re.match(line)
            if m:
                self.number_frames = int(m.group(1))
            m = endian_re.search(line)
            if m:
                if m.group(1) == "big":
                    self.bigendian = 1
                else:
                    self.bigendian = 0
            m = stagex_re.match(line)
            if m:
                self.stage_x = float(m.group(1))
            m = stagey_re.match(line)
            if m:
                self.stage_y = float(m.group(1))
            m = lock_target_re.match(line)
            if m:
                self.lock_target = float(m.group(1))
            m = scalemax_re.match(line)
            if m:
                self.scalemax = int(m.group(1))
            m = scalemin_re.match(line)
            if m:
                self.scalemin = int(m.group(1))

        # set defaults, probably correct, but warn the user
        # that they couldn't be determined from the inf file.
        if not self.image_height:
            print("Could not determine image size, assuming 256x256.")
            self.image_height = 256
            self.image_width = 256

    def load_frame(self, frame_number):
        """
        Load a frame & return it as a np array.
        """
        super(DaxReader, self).load_frame(frame_number)

        startByte = frame_number * self.image_height * self.image_width * 2
        endByte = startByte + 2*(self.image_height * self.image_width)

        dataFormat = np.dtype('uint16')
        if self.bigendian:
            dataFormat = dataFormat.newbyteorder('>')

        image_data = np.frombuffer(
            self._filePortal.read_file_bytes(startByte, endByte),
            dtype=dataFormat)
        image_data = np.reshape(image_data,
                                [self.image_height, self.image_width])
        return image_data

