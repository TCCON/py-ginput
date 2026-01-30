#!/usr/bin/env bash

# mod/vmr/map/map.nc files
cp -fv test_output_data/mod_files/fpit/oc/vertical/FPIT_20180101* test_input_data/mod_files/fpit/oc/vertical/
cp -fv test_output_data/mod_files/it/oc/vertical/IT_20250302* test_input_data/mod_files/it/oc/vertical/
cp -fv test_output_data/vmr_files/fpit/JL1_20180101* test_input_data/vmr_files/fpit/
cp -fv test_output_data/vmr_files/it/JL1_20250302* test_input_data/vmr_files/it/
cp -fv test_output_data/map_files/fpit/oc_37N_097W_20180101* test_input_data/map_files/fpit/

# NOAA interpolation test files
cp -fv test_output_data/noaa-interp-extrap/expected/default-data/post1.6/*.nc test_input_data/noaa-interp-extrap/expected/default-data/post1.6/
cp -fv test_output_data/noaa-interp-extrap/expected/default-data/pre1.6/*.nc test_input_data/noaa-interp-extrap/expected/default-data/pre1.6/
cp -fv test_output_data/noaa-interp-extrap/expected/gap-tests/mlo-and-smo/*.nc test_input_data/noaa-interp-extrap/expected/gap-tests/mlo-and-smo/
cp -fv test_output_data/noaa-interp-extrap/expected/gap-tests/smo-only/*.nc test_input_data/noaa-interp-extrap/expected/gap-tests/smo-only/
cp -fv test_output_data/noaa-interp-extrap/expected/interp-cutoff/*.nc test_input_data/noaa-interp-extrap/expected/interp-cutoff/
cp -fv test_output_data/noaa-interp-extrap/expected/real-smo-gap/*.nc test_input_data/noaa-interp-extrap/expected/real-smo-gap/

# Satellite priors files
cp -fv test_output_data/large-files/oco/oco2_priors_56742a_250302.h5 test_input_data/large-files/oco/
cp -fv test_output_data/large-files/gosat/empty/acos_priors_090430.h5 test_input_data/large-files/gosat/empty/
cp -fv test_output_data/large-files/gosat/nonempty/acos_priors_120620.h5 test_input_data/large-files/gosat/nonempty/
