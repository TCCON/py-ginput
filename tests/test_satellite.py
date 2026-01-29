from datetime import datetime
from ginput.__main__ import main
from ginput.common_utils import mod_utils
from ginput.common_utils.test_utils import simple_xco2_calc
import h5py
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest


@pytest.mark.glacial
@pytest.mark.slow
def test_oco2_priors(
    geos_3d_met_files_by_datetime,
    oco_file_dir,
    oco_file_out_dir,
    smo_real_gap_input_dir,
    test_plots_dir
):
    """Run a test to generate an OCO-2 priors file and compare against a benchmark.

    In case you need to re-run the comparison without generating the .h5 file again
    (because that is extremely slow), set the GINPUT_TEST_SKIP_GEN environmental
    variable to 1."""
    skip_gen = os.getenv('GINPUT_TEST_SKIP_GEN', '0')
    met_file = oco_file_dir / 'oco2_L2MetND_56742a_250302.h5'
    out_prior_file = oco_file_out_dir / 'oco2_priors_56742a_250302.h5'

    if skip_gen != '1':
        # This test will use the command line interface, since that is
        # how OCO ops interacts with this code.
        geos_dates = [
            datetime(2025,3,2,15),
            datetime(2025,3,2,18),
            datetime(2025,3,2,21),
        ]
        _generate_prior_h5_file(
            instrument='oco',
            met_file=met_file,
            out_prior_file=out_prior_file,
            geos_dates=geos_dates,
            geos_3d_met_files_by_datetime=geos_3d_met_files_by_datetime,
            mlo_smo_input_dir=smo_real_gap_input_dir
        )
    elif not out_prior_file.exists():
        raise FileNotFoundError(f'Output file {out_prior_file} does not exist. (GINPUT_TEST_SKIP_GEN was set to {skip_gen}, causing file generation to be skipped.)')

    expected_file = oco_file_dir / 'oco2_priors_56742a_250302.h5'
    plot_file = test_plots_dir / 'test_oco2_priors.png'
    compare_prior_h5_files(
        expected_file=expected_file,
        current_file=out_prior_file,
        met_file=met_file,
        plot_file=plot_file
    )


@pytest.mark.glacial
@pytest.mark.slow
def test_gosat_empty_priors(
    geos_3d_met_files_by_datetime,
    gosat_empty_file_dir,
    gosat_empty_file_out_dir,
    smo_real_gap_input_dir,
    test_plots_dir,
):
    """Tests that a GOSAT priors file is generated even when there are no valid soundings.

    This is more likely with GOSAT than OCO-2/3 because of the limited number of soundings.
    This was a test case that CSU ran into during ACOS B11 processing that caused a crash.
    """
    skip_gen = os.getenv('GINPUT_TEST_SKIP_GEN', '0')
    met_file = gosat_empty_file_dir / 'acos_L2Met_090430_15_B11_20250418191958.h5'
    out_prior_file = gosat_empty_file_out_dir / 'acos_priors_090430.h5'

    if skip_gen != '1':
        # This test will use the command line interface, since that is
        # how most users ops interact with this code.
        geos_dates = [
            datetime(2009,4,30,6),
            datetime(2009,4,30,9),
            datetime(2009,4,30,12),
        ]
        _generate_prior_h5_file(
            instrument='acos',
            met_file=met_file,
            out_prior_file=out_prior_file,
            geos_dates=geos_dates,
            geos_3d_met_files_by_datetime=geos_3d_met_files_by_datetime,
            mlo_smo_input_dir=smo_real_gap_input_dir
        )
    elif not out_prior_file.exists():
        raise FileNotFoundError(f'Output file {out_prior_file} does not exist. (GINPUT_TEST_SKIP_GEN was set to {skip_gen}, causing file generation to be skipped.)')

    expected_file = gosat_empty_file_dir / 'acos_priors_090430.h5'
    plot_file = test_plots_dir / 'test_acos_priors_empty.png'
    compare_prior_h5_files(
        expected_file=expected_file,
        current_file=out_prior_file,
        met_file=met_file,
        instrument='gosat',
        plot_file=plot_file
    )


@pytest.mark.glacial
@pytest.mark.slow
def test_gosat_nonempty_priors(
    geos_3d_met_files_by_datetime,
    gosat_nonempty_file_dir,
    gosat_nonempty_file_out_dir,
    smo_real_gap_input_dir,
    test_plots_dir,
):
    """Tests that the GOSAT priors for one test granule match a benchmark.
    """
    skip_gen = os.getenv('GINPUT_TEST_SKIP_GEN', '0')
    met_file = gosat_nonempty_file_dir / 'gosat_resampler.h5'
    out_prior_file = gosat_nonempty_file_out_dir / 'acos_priors_120620.h5'

    if skip_gen != '1':
        # This test will use the command line interface, since that is
        # how most users interact with this code.
        geos_dates = [
            datetime(2012,6,20,0),
            datetime(2012,6,20,3),
        ]
        _generate_prior_h5_file(
            instrument='acos',
            met_file=met_file,
            out_prior_file=out_prior_file,
            geos_dates=geos_dates,
            geos_3d_met_files_by_datetime=geos_3d_met_files_by_datetime,
            mlo_smo_input_dir=smo_real_gap_input_dir
        )
    elif not out_prior_file.exists():
        raise FileNotFoundError(f'Output file {out_prior_file} does not exist. (GINPUT_TEST_SKIP_GEN was set to {skip_gen}, causing file generation to be skipped.)')

    expected_file = gosat_nonempty_file_dir / 'acos_priors_120620.h5'
    plot_file = test_plots_dir / 'test_acos_priors_nonempty.png'
    compare_prior_h5_files(
        expected_file=expected_file,
        current_file=out_prior_file,
        met_file=met_file,
        instrument='gosat',
        plot_file=plot_file
    )


def _generate_prior_h5_file(
    instrument,
    met_file,
    out_prior_file,
    geos_dates,
    geos_3d_met_files_by_datetime,
    mlo_smo_input_dir,
):
    # This test will use the command line interface, since that is
    # how ops interacts with this code.
    geos_files = ','.join(str(geos_3d_met_files_by_datetime[d]) for d in geos_dates)

    nprocs = os.getenv('GINPUT_TEST_NPROCS', '8')
    cli_args = [
        instrument,
        '--verbose',
        '--mlo-co2-file', str(mlo_smo_input_dir / 'co2_mlo_monthly.txt'),
        '--smo-co2-file', str(mlo_smo_input_dir / 'co2_smo_monthly.txt'),
        '--truncate-mlo-smo-by', '1',
        '--fo2-file', str(mlo_smo_input_dir / 'o2_dmf_yearly_2024.txt'),
        '--nprocs', nprocs,
        geos_files,
        str(met_file),
        str(out_prior_file)
    ]

    main(cli_args)

def compare_prior_h5_files(expected_file, current_file, met_file, instrument='oco', plot_file=None):
    expected_data = _load_prior_h5_file(expected_file, met_file, instrument=instrument)
    current_data = _load_prior_h5_file(current_file, met_file, instrument=instrument)

    if plot_file is not None:
        _, axs = plt.subplots(2, 2, figsize=(12,8), gridspec_kw={'wspace': 0.4, 'hspace': 0.4})
        _plot_lat_binned_priors(expected_data=expected_data, current_data=current_data, axs=axs[0])
        _plot_prior_xco2_vs_lat(expected_data=expected_data, current_data=current_data, axs=axs[1])
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()

    assert np.allclose(expected_data['lat'], current_data['lat'], equal_nan=True), 'Files have different latitudes'
    assert np.allclose(expected_data['lon'], current_data['lon'], equal_nan=True), 'Files have different longitudes'
    assert np.allclose(expected_data['co2_prior'], current_data['co2_prior'], equal_nan=True), 'Files CO2 profiles differ'


def _load_prior_h5_file(file, met_file=None, instrument='oco'):
    with h5py.File(file) as f:
        if 'CO2Prior' in f.keys():
            # OCO-style L2CPr file
            data = {
                'co2_prior': f['CO2Prior']['co2_prior_profile_cpr'][:] * 1e6,
                'pres': f['Meteorology']['vector_pressure_levels_met'][:] * 1e-2,
                'surf_pres': f['Meteorology']['surface_pressure_met'][:] * 1e-2,
                'lat': f['SoundingGeometry']['sounding_latitude'][:],
                'lon': f['SoundingGeometry']['sounding_longitude'][:],
            }
        else:
            # Assume this is the direct output from ginput
            with h5py.File(met_file) as m:
                data = {
                    'co2_prior': f['priors']['co2_prior'][:] * 1e6,
                    'pres': f['priors']['pressure'][:],
                    'surf_pres': m['Meteorology']['surface_pressure_met'][:] * 1e-2,
                    'lat': f['priors']['sounding_latitude'][:],
                    'lon': f['priors']['sounding_longitude'][:],
                }

        for v in data.values():
            v[v < -999.0] = np.nan

        if instrument == 'gosat':
            # GOSAT will give 2D prior and pressure, 1D lat and lon, and 3D surface pressure.
            # To be consistent with OCO, that needs to be 3D prior and pressure (with the vertical
            # dimension last), 2D lat and lon, and 2D surface pressure. The second dimension of
            # surface pressure needs cut down to 1 to match our lat/lon.
            data['co2_prior'] = data['co2_prior'][:, np.newaxis, :]
            data['pres'] = data['pres'][:, np.newaxis, :]
            data['surf_pres'] = data['surf_pres'][:, :1, 0]
            data['lat'] = data['lat'][:, np.newaxis]
            data['lon'] = data['lon'][:, np.newaxis]
        elif instrument != 'oco':
            raise NotImplementedError(f'instrument = {instrument}')

        return data


def _plot_lat_binned_priors(expected_data, current_data, axs=None):
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(12,4), sharey=True)

    handles = [
        Line2D([0], [0], color='black', ls='--', marker='x', label='Expected'),
        Line2D([0], [0], color='black', ls=':', marker='+', label='Current'),
    ]
    tmp_h = _plot_lat_binned_priors_inner(
        lats=expected_data['lat'],
        priors=expected_data['co2_prior'],
        pres=expected_data['pres'],
        ls='--', mkr='x', ax=axs[0]
    )
    _plot_lat_binned_priors_inner(
        lats=current_data['lat'],
        priors=current_data['co2_prior'],
        pres=current_data['pres'],
        ls=':', mkr='+', ax=axs[0]
    )

    curr_shp = current_data['co2_prior'].shape
    exp_shp = expected_data['co2_prior'].shape
    if curr_shp == exp_shp:
        _plot_lat_binned_priors_inner(
            lats=current_data['lat'],
            priors=current_data['co2_prior'] - expected_data['co2_prior'],
            pres=current_data['pres'],
            ls='-', mkr='.', ax=axs[1]
        )
    else:
        axs[1].text(0.5, 0.5, f'Current shape {curr_shp} differs from\nexpected shape {exp_shp}',
                    transform=axs[1].transAxes, ha='center', va='center')
    handles.extend(tmp_h)
    axs[1].legend(handles=handles, loc='center left', bbox_to_anchor=(1.025, 0.5))
    axs[0].set_xlabel('Prior CO$_2$ (ppm)')
    axs[1].set_xlabel('$\\Delta$ Prior CO$_2$ (current - expected, ppm)')
    for ax in axs:
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_ylabel('Pressure (hPa)')


def _plot_lat_binned_priors_inner(lats, priors, pres, ls, mkr, ax):
    nlev = priors.shape[-1]
    lats = lats.ravel()
    priors = priors.reshape(-1, nlev)
    pres = pres.reshape(-1, nlev)
    colors = plt.get_cmap('tab10')
    lat_bin_edges = np.arange(-90.0, 91.0, 20.0)

    ibin = -1
    bin_dummy_lines = []
    for y1, y2 in zip(lat_bin_edges[:-1], lat_bin_edges[1:]):
        ibin += 1
        yy = (lats >= y1) & (lats < y2)
        if yy.sum() == 0:
            continue

        mean_prior = np.nanmean(priors[yy], axis=0)
        mean_pres = np.nanmean(pres[yy], axis=0)

        y1s = mod_utils.format_lat(y1, prec=0)
        y2s = mod_utils.format_lat(y2, prec=0)
        ax.plot(mean_prior, mean_pres, color=colors(ibin), ls=ls, marker=mkr)
        bin_dummy_lines.append(
            Line2D([0], [0], color=colors(ibin), label=f'{y1s} to {y2s}')
        )
    ax.grid(True)
    return bin_dummy_lines


def _plot_prior_xco2_vs_lat(expected_data, current_data, axs=None):
    if axs is None:
        _, axs = plt.subplots(1, 2, figsize=(12,4))

    expected_xco2 = simple_xco2_calc(expected_data['co2_prior'], pressure_levels=expected_data['pres'], surface_pressure=expected_data['surf_pres'])
    current_xco2 = simple_xco2_calc(current_data['co2_prior'], pressure_levels=current_data['pres'], surface_pressure=current_data['surf_pres'])

    ax = axs[0]
    ax.plot(expected_data['lat'].ravel(), expected_xco2.ravel(), marker='x', ls='none', label='Expected')
    ax.plot(current_data['lat'].ravel(), current_xco2.ravel(), marker='+', ls='none', label='Current')
    ax.legend(ncol=2, loc='upper left')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('XCO$_2$ (ppm)')
    ax.grid(True)

    curr_shp = current_xco2.shape
    exp_shp = expected_xco2.shape
    ax = axs[1]
    if curr_shp == exp_shp:

        ax.plot(current_data['lat'].ravel(), current_xco2.ravel() - expected_xco2.ravel(), marker='.', ls='none')
        ax.set_ylabel(r'$\Delta$XCO$_2$ (current - expected, ppm)')
        ax.set_xlabel('Latitude')
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, f'Current shape {curr_shp} differs from\nexpected shape {exp_shp}', ha='center', va='center', transform=ax.transAxes)
