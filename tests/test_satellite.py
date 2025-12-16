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
    skip_gen = os.getenv('GINPUT_TEST_SKIP_GEN', '0')
    met_file = oco_file_dir / 'oco2_L2MetND_56742a_250302.h5'
    out_prior_file = oco_file_out_dir / 'oco2_priors_56742a_250302.h5'

    import pdb
    pdb.set_trace()
    if skip_gen != '1':
        # This test will use the command line interface, since that is
        # how OCO ops interacts with this code.
        geos_dates = [
            datetime(2025,3,2,15),
            datetime(2025,3,2,18),
            datetime(2025,3,2,21),
        ]
        geos_files = ','.join(str(geos_3d_met_files_by_datetime[d]) for d in geos_dates)

        nprocs = os.getenv('GINPUT_TEST_NPROCS', '8')
        cli_args = [
            'oco',
            '--verbose',
            '--mlo-co2-file', str(smo_real_gap_input_dir / 'co2_mlo_monthly.txt'),
            '--smo-co2-file', str(smo_real_gap_input_dir / 'co2_smo_monthly.txt'),
            '--truncate-mlo-smo-by', '1',
            '--fo2-file', str(smo_real_gap_input_dir / 'o2_dmf_yearly_2024.txt'),
            '--nprocs', nprocs,
            geos_files,
            str(met_file),
            str(out_prior_file)
        ]

        main(cli_args)
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


def compare_prior_h5_files(expected_file, current_file, met_file, plot_file=None):
    expected_data = _load_prior_h5_file(expected_file, met_file)
    current_data = _load_prior_h5_file(current_file, met_file)

    if plot_file is not None:
        _, axs = plt.subplots(2, 2, figsize=(12,8), gridspec_kw={'wspace': 0.4, 'hspace': 0.4})
        _plot_lat_binned_priors(expected_data=expected_data, current_data=current_data, axs=axs[0])
        _plot_prior_xco2_vs_lat(expected_data=expected_data, current_data=current_data, axs=axs[1])
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()

    assert np.allclose(expected_data['lat'], current_data['lat'], equal_nan=True), 'Files have different latitudes'
    assert np.allclose(expected_data['lon'], current_data['lon'], equal_nan=True), 'Files have different longitudes'
    assert np.allclose(expected_data['co2_prior'], current_data['co2_prior'], equal_nan=True), 'Files CO2 profiles differ'


def _load_prior_h5_file(file, met_file=None):
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
