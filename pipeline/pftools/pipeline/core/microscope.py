import json
import yaml
import os
from typing import Optional
class MicroscopeInformation:
    """
    Microscope information has JSON file with the following:
        flip_horizontal: bool
        flip_vertical: bool
        transpose: bool 
        microns_per_pixel: float # microns per pixel in x and y
        image_dimensions: [int, int] # number of pixels in x and y
        mag: int # objective magnification
        tubelens: int # tube lens length in mm (assuming Nikon 200 mm tube lens is standard)
        na: float # objective numerical aperture 
        ni: float # immersion medium refractive index
        wd: float # working distance in um
        z_step: float # z-step in um
    """
    def __init__(self, microscope_path: Optional[str]=None):
        self._microscope_path = microscope_path
        if microscope_path is not None and os.path.exists(microscope_path):
            with open(microscope_path, 'r') as f:
                self._microscope_information = json.load(f)
        else:
            self._microscope_information = {}
        self.flip_horizontal = self._microscope_information.get('flip_horizontal', True) # True to match MERlin
        self.flip_vertical = self._microscope_information.get('flip_vertical', False)
        self.transpose = self._microscope_information.get('transpose', True)
        self.microns_per_pixel = self._microscope_information.get('microns_per_pixel', 0.108) # MERlin default -- 60X with Hamamatsu ORCA
        self.image_dimensions = self._microscope_information.get('image_dimensions', [2048, 2048]) # MERlin default -- 60X with Hamamatsu ORCA
        self.mag = self._microscope_information.get('mag', 60)
        self.tubelens = self._microscope_information.get('tubelens', 200)
        self.na = self._microscope_information.get('na', 1.4)
        self.ni = self._microscope_information.get('ni', 1.515)
        self.wd = self._microscope_information.get('wd', 150)
        self.z_step = self._microscope_information.get('z_step', 1.5)

    def save_microscope_yaml(self, microscope_path: str):
        """ Save the microscope information to a JSON file.

        Args:
            microscope_path: The path to the file to save the microscope information to.
        """
        with open(microscope_path, 'w') as f:
            yaml.dump(self._microscope_information, f, indent=4, sort_keys=True)