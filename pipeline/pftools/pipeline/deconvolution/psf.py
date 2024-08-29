import numpy as np
from pftools.pipeline.deconvolution.microscope_psf import MicroscopePSF

# models of PSFs
def nikon25x105na(xy_size=17, z_size=17, dxy=0.173, dz=0.5, na_number=1.05, wvl=0.561, tl=300.0):
    return generate_psf_custom(
         dxy=dxy, dz=dz, xy_size=xy_size, z_size=z_size, wvl=wvl, NA=na_number, tl=tl*1.0e3
    )

def nikon40x125na(xy_size=17, z_size=17, dxy=0.11, dz=0.75, NA=1.25, wvl=0.561, tl=300.0):
    return generate_psf_custom(
         dxy=dxy, dz=dz, xy_size=xy_size, z_size=z_size, M=40, NA=NA, n=1.405, wd=300, tl=tl * 1.0e3, wvl=wvl
    )

def nikon60x14na(xy_size=25, z_size=25, dxy=0.11, dz=0.25, NA=1.4, wvl=0.561, tl=200.0):
    return generate_psf_custom(
         dxy=dxy, dz=dz, xy_size=xy_size, z_size=z_size, M=40, NA=NA, n=1.51, wd=150, tl=tl * 1.0e3, wvl=wvl
    )


def generate_psf_custom(dxy, dz, xy_size, z_size, M=25, NA=1.05, n=1.33, wd=550, tl=300.0 * 1.0e3, wvl=0.561, ni=1.405):
    """
    Generates a 3D PSF array.
    :param dxy: voxel dimension along xy (microns)
    :param dz: voxel dimension along z (microns)
    :param xy_size: size of PSF kernel along x and y (odd integer)
    :param z_size: size of PSF kernel along z (odd integer)
        self.parameters = {
            "M": 100.0,  # magnification
            "NA": 1.4,  # numerical aperture
            "ng0": 1.515,  # coverslip RI design value
            "ng": 1.515,  # coverslip RI experimental value
            "ni0": 1.515,  # immersion medium RI design value
            "ni": 1.515,  # immersion medium RI experimental value
            "ns": 1.33,  # specimen refractive index (RI)
            "ti0": 150,  # microns, working distance (immersion medium thickness) design value
            "tg": 170,  # microns, coverslip thickness experimental value
            "tg0": 170,  # microns, coverslip thickness design value
            "zd0": 200.0 * 1.0e3,
        }  # microscope tube length (in microns).

    """

    psf_gen = MicroscopePSF()

    # Microscope parameters.
    psf_gen.parameters["M"] = M  # magnification
    psf_gen.parameters["NA"] = NA  # numerical aperture
    psf_gen.parameters["ni0"] = ni
    psf_gen.parameters["ni"] = ni
    psf_gen.parameters["ns"] = n
    psf_gen.parameters["ti0"] = wd
    psf_gen.parameters["zd0"] = tl

    lz = (z_size) * dz
    z_offset = -(lz - 2 * dz) / 2
    pz = np.arange(0, lz, dz)

    # gLXYZParticleScan(self, dxy, xy_size, pz, normalize = True, wvl = 0.6, zd = None, zv = 0.0):
    psf_xyz_array = psf_gen.gLXYZParticleScan(dxy=dxy, xy_size=xy_size, pz=pz, zv=z_offset, wvl=wvl)

    psf_xyz_array /= psf_xyz_array.sum()

    #aprint(f"Generating PSF for parameters: {psf_gen.parameters}")

    return psf_xyz_array
