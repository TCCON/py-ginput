from enum import Enum
import netCDF4 as ncdf
import os
import re

class GeosSource(Enum):
    FPIT = 'fpit'
    FP = 'fp'
    IT = 'it'
    UNKNOWN = 'UNKNOWN'

    @classmethod
    def from_str(cls, s):
        if s in {'fpit', 'fp', 'it'}:
            return cls(s)
        else:
            return cls.UNKNOWN

class GeosVersion:
    def __init__(self, version_str: str, source: str | GeosSource):
        self.version_str = version_str
        if isinstance(source, str):
            self.source = GeosSource.from_str(source)
        else:
            self.source = source

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            return self.version_str == other.version_str and self.source == other.source
            
    def __repr__(self):
        clsname = self.__class__.__name__
        return f'<{clsname}({self.source}, {self.version_str})>'
            
    def __str__(self):
        return f'{self.source.value} (GEOS v{self.version_str})'
    
    @classmethod
    def from_str(cls, s):
        m = re.match(r'(\w+) \(GEOS v(\d+\.\d+\.\d+)\)', s)
        if m is None:
            raise ValueError(f'"{s}" does not match the expected format of a ginput GEOS version string')
        else:
            vstr = m.group(2)
            src_str = m.group(1)
            return cls(vstr, src_str)
        
    @classmethod
    def from_nc_dset(cls, nc_dset):
        vstr = nc_dset.VersionID
        granule = nc_dset.GranuleID
        # Assume a granule name like "GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2008-01-01T0000.V01.nc4"
        # where the "source" is the second component as divided by "."s.
        source_str = granule.split('.')[1]
        return cls(vstr, source_str)
    
    @classmethod
    def from_nc_file(cls, nc_file):
        with ncdf.Dataset(nc_file) as ds:
            return cls.from_nc_dset(ds)


avogadro = 6.022141e23  # molec./mol
mass_dry_air = 28.964e-3  # kg/mol
mass_h2o = 18.01534e-3  # kg/mol
gas_const = 8.314e4 # gas constant in cm^3 * hPa / (mol * K)

ratio_molec_mass = 28.964/18.02	 # Ratio of Molecular Masses (Dry_Air/H2O)

trop_flag = 1
middleworld_flag = 2
overworld_flag = 3

_mydir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(_mydir, '..', 'data'))

days_per_year = 365.25

priors_version = '1.0.0'

# US Standard Atmosphere
p_ussa = (10.0,  5.0,   2.0,   1.0,   0.5,    0.2,   0.1,   0.01,  0.001, 0.0001)
t_ussa = (227.7, 239.2, 257.9, 270.6, 264.3, 245.2, 231.6, 198.0, 189.8, 235.0)
z_ussa = (31.1,  36.8,  42.4,  47.8,  53.3,  60.1,  64.9,  79.3,  92.0,  106.3)
