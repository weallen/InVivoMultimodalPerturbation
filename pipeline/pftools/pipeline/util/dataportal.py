import os
from abc import abstractmethod, ABC
from typing import List
from time import sleep


class DataPortal(ABC):

    """
    A superclass for reading files within a specified directory from a
    data storage service.
    """

    def __init__(self, basePath: str):
        super().__init__()

        self._basePath = basePath

    @staticmethod
    def create_portal(basePath: str) -> 'DataPortal':
        """ Create a new portal capable of reading from the specified basePath.
        Args:
            basePath: the base path of the data portal
        Returns: a new DataPortal for reading from basePath
        """
        return LocalDataPortal(basePath)

    @abstractmethod
    def is_available(self) -> bool:
        """ Determine if the basePath represented by this DataPortal is
        currently accessible.
        Returns: True if the basePath is available, otherwise False.
        """
        pass

    @abstractmethod
    def open_file(self, fileName: str) -> 'FilePortal':
        """ Open a reference to a file within the basePath represented
        by this DataPortal.
        Args:
            fileName: the name of the file to open, relative to basePath
        Returns: A FilePortal referencing the specified file.
        """
        pass

    @staticmethod
    def _filter_file_list(inputList: List[str], extensionList: List[str]
                          ) -> List[str]:
        if not extensionList:
            return inputList
        return [f for f in inputList if any(
            [f.endswith(x) for x in extensionList])]

    @abstractmethod
    def list_files(self, extensionList: List[str] = None) -> List[str]:
        """ List all the files within the base path represented by this
        DataReader.
        Args:
            extensionList: a list of extensions of files to filter for. Only
                files ending in one of the extensions will be returned.
        Returns: a list of the file paths
        """
        pass

class FilePortal(ABC):

    """
    A superclass for reading a specified file from a data storage service.
    """

    def __init__(self, fileName: str):
        super().__init__()
        self._fileName = fileName

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        self.close()

    def get_file_name(self) -> str:
        """ Get the name of the file accessed by this file portal.
        Returns: The full file name
        """
        return self._fileName

    def get_file_extension(self) -> str:
        """ Get the extension of the file accessed by this file portal.
        Returns: The file extension
        """
        return os.path.splitext(self._fileName)[1]

    def _exchange_extension(self, newExtension: str) -> str:
        return ''.join([os.path.splitext(self._fileName)[0], newExtension])

    @abstractmethod
    def exists(self) -> bool:
        """ Determine if this file exists within the dataset.
        Returns: Flag indicating whether or not the file exists
        """
        pass

    @abstractmethod
    def get_sibling_with_extension(self, newExtension: str) -> 'FilePortal':
        """ Open the file with the same base name as this file but with the
        specified extension.
        Args:
            newExtension: the new extension
        Returns: A reference to the file with the extension exchanged.
        """
        pass

    @abstractmethod
    def read_as_text(self) -> str:
        """ Read the contents of this file as a string.
        Returns: the file contents as a string
        """
        pass

    @abstractmethod
    def read_file_bytes(self, startByte: int, endByte: int) -> bytes:
        """ Read bytes within the specified range from this file.
        Args:
            startByte: the index of the first byte to read (inclusive)
            endByte: the index at the end of the range of bytes to read
                (exclusive)
        Returns: The bytes between startByte and endByte within this file.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """ Close this file portal."""
        pass


class LocalFilePortal(FilePortal):

    """
    A file portal for accessing a file in a local file system.
    """

    def __init__(self, fileName: str):
        super().__init__(fileName)
        self._fileHandle = open(fileName, 'rb')

    def get_sibling_with_extension(self, newExtension: str):
        return LocalFilePortal(self._exchange_extension(newExtension))

    def exists(self):
        return os.path.exists(self._fileName)

    def read_as_text(self):
        self._fileHandle.seek(0)
        return self._fileHandle.read().decode('utf-8')

    def read_file_bytes(self, startByte, endByte):
        self._fileHandle.seek(startByte)
        return self._fileHandle.read(endByte-startByte)

    def close(self) -> None:
        self._fileHandle.close()

class LocalDataPortal(DataPortal):

    """
    A class for accessing data that is stored in a local file system.
    """

    def __init__(self, basePath: str):
        super().__init__(basePath)

    def is_available(self):
        return os.path.exists(self._basePath)

    def open_file(self, fileName):
        if os.path.abspath(self._basePath) in os.path.abspath(fileName):
            return LocalFilePortal(fileName)
        else:
            return LocalFilePortal(os.path.join(self._basePath, fileName))

    def list_files(self, extensionList=None):
        allFiles = [os.path.join(self._basePath, currentFile)
                    for currentFile in os.listdir(self._basePath)]
        return self._filter_file_list(allFiles, extensionList)
