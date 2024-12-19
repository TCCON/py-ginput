#!/bin/bash

conda_env=$(basename $CONDA_PREFIX)
if [[ ! -z $FORCE_INSTALL ]] && [[ $FORCE_INSTALL == "1" ]]; then
  echo "Installing into $conda_env"
elif [[ $conda_env != "ginput-auto-default" ]]; then
  echo "Error: current environment is not ginput-auto-default, call with FORCE_INSTALL=1 to allow installing in the current environment"
  exit 2
fi

which dot >& /dev/null
if [[ $? != 0 ]]; then
  echo "Warning: dot is not installed or not on your path. Some graphs in the documentation will not build. Please install graphviz with your package manager."
fi

if [[ $GGG_USE_MICROMAMBA == "1" ]]; then
  CMD=micromamba
else
  CMD=conda
fi

${CMD} install -c conda-forge sphinx sphinx_rtd_theme myst-parser linkify-it-py
