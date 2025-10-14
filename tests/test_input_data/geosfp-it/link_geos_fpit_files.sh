#!/bin/bash

function usage() {
    cat << EOF
USAGE:
    $0 GEOS_FPIT_PATH
    $0 MET_PATH CHM_PATH
    $0 3D_MET_PATH 2D_MET_PATH 3D_CHM_PATH

This script will link the needed GEOS FP-IT data for testing
into this directory. If you have your data in the following
structure:

    <GEOS_FPIT_PATH>/met/Nv <-- 3d model-level met files
    <GEOS_FPIT_PATH>/met/Nx <-- 2d met files
    <GEOS_FPIT_PATH>/chm/Nv <-- 3d model-level chemical files

then you only need give the leading GEOS_FPIT_PATH as the sole
argument. 

If your meteorology and chemistry files are not in "met" and 
"chm" subdirectories of the same directory, but do have the 
expected Nv and (for met) Nx subdirectories, use the 2-argument
option and specify the met and chem paths (i.e. the directories
with Nv/Nx subdirectories).

If you do not follow this organization at all, use the three
argument form and give the paths to directories containing
the 3D model level met files, 2D met files, and 3D model level
chemistry files as the three arguments.

Note that if Nv or Nx directories exist here (in the 
test_input_data/geosfp-it directory), this may not work if the
files it needs to create already exist.

Also note that absolute paths work best for these arguments;
if you give relative paths, they must work from directories
"$(dirname $0)/Nv" and "$(dirname $0)/Nx".
EOF
}


if [[ $# -lt 1 ]] || [[ $1 == '-h' ]] || [[ $1 == '--help' ]]; then
    usage
    exit 2
elif [[ $# -eq 1 ]]; then
    MET_3D="$1/met/Nv"
    MET_2D="$1/met/Nx"
    CHM_3D="$1/chm/Nv"
elif [[ $# -eq 2 ]]; then
    MET_3D="$1/Nv"
    MET_2D="$1/Nx"
    CHM_3D="$2/Nv"
else
    MET_3D="$1"
    MET_2D="$2"
    CHM_3D="$3"   
fi


missing=0
if [[ ! -d $MET_3D ]]; then
    2>&1 echo "ERROR: Met 3d directory ($MET_3D) does not exist or is not a directory."
    missing=1
fi
if [[ ! -d $MET_2D ]]; then
    2>&1 echo "ERROR: Met 2d directory ($MET_2D) does not exist or is not a directory."
    missing=1
fi
if [[ ! -d $CHM_3D ]]; then
    2>&1 echo "ERROR: Chem 3d directory ($CHM_3D) does not exist or is not a directory."
    missing=1
fi

if [[ $missing -eq 1 ]]; then
    exit 1
fi

mydir="$(dirname $0)"

echo "Linking 3D met files from $MET_3D"
echo "Linking 2D met files from $MET_2D"
echo "Linking 3D chem files from $CHM_3D"

mkdir -pv "$mydir"/Nv
mkdir -pv "$mydir"/Nx
ln -sv "$MET_3D"/GEOS.fpit.asm.inst3_3d_asm_Nv.GEOS5124.20180101_{00,03,06,09,12,15,18,21}00.V01.nc4 "$mydir"/Nv
ln -sv "$CHM_3D"/GEOS.fpit.asm.inst3_3d_chm_Nv.GEOS5124.20180101_{00,03,06,09,12,15,18,21}00.V01.nc4 "$mydir"/Nv
ln -sv "$MET_2D"/GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20180101_{00,03,06,09,12,15,18,21}00.V01.nc4 "$mydir"/Nx

