from argparse import ArgumentParser
from collections import deque
import h5py
from pathlib import Path

OCO_KEEP_MET_VARS = [
    '/Meteorology/epv_profile_met',
    '/Meteorology/temperature_profile_met',
    '/Meteorology/vector_pressure_levels_met',
    '/Meteorology/surface_pressure_met',
    '/SoundingGeometry/sounding_time_string',
    '/Meteorology/height_profile_met',
    '/SoundingGeometry/sounding_latitude',
    '/SoundingGeometry/sounding_longitude',
    '/Meteorology/blended_tropopause_pressure_met',
    '/Meteorology/tropopause_temperature_met',
    '/Meteorology/gph_met',
    '/SoundingGeometry/sounding_qual_flag',
]

def main():
    p = ArgumentParser(description='Subset test satellite granules for faster integration tests')
    p.add_argument('to_subset', choices=['all', 'oco'], help="Which satellite's test granules to subset")
    p.add_argument('--pdb', action='store_true', help='Launch Python debugger')
    clargs = p.parse_args()
    to_subset = clargs.to_subset

    if clargs.pdb:
        import pdb
        pdb.set_trace()

    out_dir = Path(__file__).parent
    if to_subset in {'all', 'oco'}:
        orig_met = Path('../large-files/oco/oco2_L2MetND_56742a_250302.h5')
        orig_priors = Path('../large-files/oco/oco2_priors_56742a_250302.h5')
        # Frame index 0 has no fill values, 4465 has a single sounding that
        # is fills, and 4943 a full frame of fill values. (Checked in my
        # 2026-11-12 notebook on ccycle - JLL.)
        subset_file_pair(
            orig_met=orig_met, orig_priors=orig_priors, out_dir=out_dir,
            keep_met_vars=OCO_KEEP_MET_VARS,
            frame_indices=[0, 4465, 4943], frame_dim_length=8370
        )



def subset_file_pair(orig_met, orig_priors, out_dir, keep_met_vars, frame_indices, frame_dim_length):
    if not orig_met.exists():
        raise FileNotFoundError(f'Met file {orig_met} not found, were large files downloaded?')
    if not orig_priors.exists():
        raise FileNotFoundError(f'Priors file {orig_priors} not found, were large files downloaded?')

    out_met = out_dir / f'{orig_met.stem}_subset{orig_met.suffix}'
    out_priors = out_dir / f'{orig_priors.stem}_subset{orig_priors.suffix}'
    subset_h5_file(
        input_file=orig_met, output_file=out_met,
        subset_inds=frame_indices, subset_dim_length=frame_dim_length,
        vars_to_keep=keep_met_vars
    )
    subset_h5_file(
        input_file=orig_priors, output_file=out_priors,
        subset_inds=frame_indices, subset_dim_length=frame_dim_length,
        vars_to_keep=None
    )


def subset_h5_file(input_file, output_file, subset_inds, subset_dim_length, vars_to_keep=None):
    print(f'Creating {output_file}')
    with h5py.File(input_file) as inf, h5py.File(output_file, 'w') as outf:
        nskipped = 0
        nfound = 0
        ntotal = 0
        for varname, invar in _walk_h5_vars(inf):
            ntotal += 1
            if vars_to_keep is None or varname in vars_to_keep:
                if invar.shape != () and invar.shape[0] == subset_dim_length:
                    data = invar[:][subset_inds]
                    outvar = outf.create_dataset(varname, shape=data.shape, dtype=invar.dtype, data=data, fillvalue=invar.fillvalue)
                else:
                    outvar = outf.create_dataset_like(varname, invar)
                    outvar[()] = invar[()]

                for attname, attval in invar.attrs.items():
                    outvar.attrs[attname] = attval
                nfound += 1
            else:
                nskipped += 1

        if vars_to_keep is None:
            print(f'Copied {nfound} of {ntotal} variables')
        else:
            print(f'Copied {nfound} of {ntotal} variables, {len(vars_to_keep)} expected')
            print(f' -> {nskipped} variables skipped')
            if nfound < len(vars_to_keep):
                print(' -> WARNING: fewer variables were copied than expected')



def _walk_h5_vars(f: h5py.File):
    """Returns an iterator over variables name and variable pairs in an open dataset.

    The variable names will have the full path; always prefixed with "/".
    """
    items = deque([('', f)])
    while len(items) > 0:
        name, item = items.popleft()
        if isinstance(item, h5py._hl.group.Group):
            for k, v in item.items():
                items.append((f'{name}/{k}', v))
        else:
            yield f'{name}', item


if __name__ == '__main__':
    main()
