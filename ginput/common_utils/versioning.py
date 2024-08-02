from enum import Enum
import netCDF4 as ncdf
import os
import re

from .mod_utils import compute_file_checksum

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
    def __init__(self, version_str: str, source: str | GeosSource, file_name: str, md5_checksum: str):
        self.version_str = version_str
        self.file_name = file_name
        self.md5_checksum = md5_checksum
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
        return f'{self.source.value} (GEOS v{self.version_str}) : {self.file_name} : {self.md5_checksum}'
    
    @classmethod
    def from_str(cls, s):
        parts = [x.strip() for x in s.split(':')]
        if len(parts) != 3:
            raise ValueError(f'"{s}" does not contain three substrings separated by colons')
        
        version, file_name, checksum = parts
        m = re.match(r'(\w+) \(GEOS v(\d+\.\d+\.\d+)\)', version)
        if m is None:
            raise ValueError(f'"{s}" does not match the expected format of a ginput GEOS version string')
        else:
            vstr = m.group(2)
            src_str = m.group(1)
            return cls(vstr, src_str, file_name, checksum)
        
    @classmethod
    def from_nc_file(cls, nc_file):
        file_name = os.path.basename(nc_file)
        file_hash = compute_file_checksum(nc_file)
        with ncdf.Dataset(nc_file) as nc_dset:
            vstr = nc_dset.VersionID
            granule = nc_dset.GranuleID
            # Assume a granule name like "GEOS.it.asm.asm_inst_3hr_glo_L576x361_v72.GEOS5294.2008-01-01T0000.V01.nc4"
            # where the "source" is the second component as divided by "."s.
            source_str = granule.split('.')[1]
            return cls(vstr, source_str, file_name, file_hash)