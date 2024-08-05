from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
import netCDF4 as ncdf
import os
from pathlib import Path
import re
import shutil

from .mod_utils import compute_file_checksum
from .ioutils import make_dependent_file_hash

from typing import Union, Dict, Optional, Sequence

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
    def __init__(self, version_str: str, source: Union[str, GeosSource], file_name: str, md5_checksum: str):
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
        

def update_versioned_file_header(prev_file: Union[str, Path, Sequence[str]], new_data_descr: str, program_descr: str, source_files: Dict[str, Union[str, Path]],
                                 header_comment_symbol: str = '#', insert_line_index: int = 0):
    """For a file which can have data appended/updated, include a new entry in the HISTORY part of the header.

    This will return a list of header lines. If ``prev_file`` is ``None``, then the returned list will have
    a HISTORY block with an entry for the initial creation of the file. Otherwise, the returned list will have
    the header lines from ``prev_file`` with additional lines for the new change added. If there was already
    a HISTORY section in the header, the new lines are added to the end of that section. Otherwise, they will
    be inserted at the ``insert_line_index``, i.e. ``header[insert_line_index]`` will be the first new line
    added, and any lines from ``prev_header[insert_line_index:]`` will come after the new lines. In the latter
    case, the "HISTORY" and "END HISTORY" lines that denote a history section will be added as well.

    Parameters
    ----------
    prev_file
        Path to the previous version of the file or, if this is the first time this file is written,
        the list of header lines to be written to the new file (excluding the history block).

    new_data_descr
        A brief description of what data was added, this will be included in the first line of the new history
        entry.

    program_descr
        A brief description of what program added the new data, this will be included in the first line of the new
        history entry.

    source_files
        A dictionary where the keys are short descriptions of the source files used to compute the new data and
        the values are the paths to those files. Each source file will be added as a line in the new history
        section with its SHA1 hash.

    header_comment_symbol
        What symbol indicates that a line is a comment; this will be added to the beginning of new lines created
        for the header and, when reading a previous file, lines that begin with this string will be included as
        previous header lines.

    insert_line_index
        The index in the list of header lines to insert the history block if ``prev_file`` is given, but it has
        no history block in the header. This input is ignored if ``prev_file`` is ``None`` or there is a history
        block in its header.

    Returns
    -------
    header_lines
        A list of header lines that combines the previous header (read from the previous file or provided as
        the ``prev_file`` argument) and the new history section.
    """
    # Create the new history lines, including source files and (if given) the previous file
    cs = header_comment_symbol
    now = datetime.now(timezone.utc).astimezone()

    new_history = [
        f'{cs}  Added {new_data_descr} using {program_descr} at {now:%Y-%m-%d %H:%M %Z}. Source files:\n'
    ]
    if isinstance(prev_file, (str, Path)):
        new_history.append(f'{cs}    - Previous file: {prev_file} (SHA1 = {make_dependent_file_hash(prev_file)})\n')
    for file_type, file_path in source_files.items():
        new_history.append(
            f'{cs}    - {file_type}: {file_path} (SHA1 = {make_dependent_file_hash(file_path)})\n'
        )

    # If given a path to a previous file, read in its header
    if isinstance(prev_file, (str, Path)):
        with open(prev_file) as f:
            header = []
            for idx, line in enumerate(f):
                if line.startswith(header_comment_symbol):
                    header.append(line)
                else:
                    break
    else:
        header = prev_file

    # Find the end of the history block, if present.
    end_hist_index = None
    for idx, line in enumerate(header):
        if line.strip() == f'{cs} END HISTORY':
            end_hist_index = idx

    if end_hist_index is not None:
        # File already has a history block, so we want to add to its end
        insert_line_index = end_hist_index
    else:
        # File does not have a history block, so we need to include the block start/end indicator lines
        new_history.insert(0, f'{cs} HISTORY:\n')
        new_history.append(f'{cs} END HISTORY\n')

    return header[:insert_line_index] + new_history + header[insert_line_index:]


class RollingBackup(ABC):
    """Base class for various scheme of making a series of backups.
    """
    def make_rolling_backup(self, src_file: Union[str, Path], max_num_backups: Optional[int] = None) -> Path:
        """Backup a file, creating a sequence of backups with an optional maximum number of backup files.

        Parameters
        ----------
        src_file
            The file to back up

        max_num_backups
            The maximum number of backup files to keep (not counting the original). If not given, there
            will be no limit on the number of backup files.

        Returns
        -------
        dest_file
            The path to the backup created.
        """
        src_file = Path(src_file)
        prev_backups = self._existing_backups(src_file)
        self._shuffle_backups(prev_backups, max_num_backups)
        dest_file = self._name(src_file)
        shutil.copy2(str(src_file), str(dest_file))
        return dest_file

    def _shuffle_backups(self, prev_backups: Sequence[Path], max_num_backups: Optional[int]):
        """Rename and/or delete previous backups to make room in the sequence for a new file.

        The default implementation just removes enough previous backups

        Parameters
        ----------
        prev_backups
            The list of previous backup files, ordered from oldest to newest.

        max_num_backups
            The maximum number of backup files to keep (not counting the original). If not given, there
            will be no limit on the number of backup files.
        """
        if max_num_backups is None or len(prev_backups) < (max_num_backups - 1):
            return
        
        n_to_remove = len(prev_backups) - max_num_backups + 1
        for prev_backup in prev_backups[:n_to_remove]:
            os.remove(prev_backup)

    @abstractmethod
    def _name(self, src_file: Path) -> Path:
        """Return the path to copy the new backup to.
        """
        pass

    @abstractmethod
    def _existing_backups(self, src_file: Path) -> Sequence[Path]:
        """Return a list of existing backup files, ordered oldest to newest.
        """
        pass


class RollingBackupByDate(RollingBackup):
    """A rolling backup implementation that appends ".bak{DATE}" to backup file name

    Parameters
    ----------
    date_fmt
        A format string for :func:`datetime.strftime` that will be used to format the
        current time when naming the backup file.
    """
    def __init__(self, date_fmt: str = '%Y%m%dT%H%M%S'):
        self._date_fmt = date_fmt

    def _name(self, src_file: Path) -> Path:
        curr_name = src_file.name
        date_str = datetime.now().strftime(self._date_fmt)
        new_name = f'{curr_name}.bak{date_str}'
        return src_file.with_name(new_name)
    
    def _existing_backups(self, src_file: Path) -> Sequence[Path]:
        src_dir = src_file.parent
        pattern = f'{src_file.name}.bak*'
        return sorted(src_dir.glob(pattern))
