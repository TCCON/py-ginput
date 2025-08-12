#!/bin/bash -l

# Installs ginput into a new Conda environment. Usage: ./install.sh [env_name]. If env_name is not specified,
# then tries to install in the active Python environment. 

if [[ $# == 0 ]] || [[ $1 == -h ]] || [[ $1 == --help ]]; then
cat << EOF
usage: $0 ENVNAME

Installs ginput and its dependencies into a conda environment.

Creates a conda environment with the name ENVNAME and installs the
packages listed in environment.yml into it, then installs ginput
in develop mode. Alternatively, if the given environment already
exists, update it with the requirements in environment.yml.

Upon successful completion, a run_ginput.py script is created in this
directory. Executing this script as ./run_ginput.py will always use
the Python that ginput was installed with.

Example: install into "ginput-py3" environment
$0 ginput-py3
EOF

exit 0
fi

envname="$1"

# Does conda exist. Can't just test `which conda` because
# some versions of conda create a shell function, not an 
# actual executable
conda -h >& /dev/null
if [[ $? != 0 ]]; then
    echo "Cannot create a conda environment for ginput because 'conda' is not on your PATH."
    echo "Install Miniforge (https://github.com/conda-forge/miniforge), or another package that provides 'conda', "
    echo "or ensure that conda is properly initialized in your shell."
    exit 1
fi

# Set the script to exit if any errors occur. We cannot do this 
# before the "conda -h >& /dev/null" check because we manually
# check if that returns a non-zero exit
set -e

# Does the environment already exist
# Make an array of env names
conda_envs=($(conda env list | grep -v '^#' | awk '{print $1}'))
env_exists=false
for e in ${conda_envs[*]}; do
    if [[ $e == $envname ]]; then
        env_exists=true
        break
    fi
done

if $env_exists; then
    echo "$envname already exists, will update installed packages"
    conda env update --name "$envname" --file environment-py312.yml
else
    echo "Will create conda environment '$envname'"
    conda env create --name "$envname" --file environment-py312.yml
fi


# the next two lines are needed to allow "conda activate" to work in a non-interactive shell
conda_base=$(conda info --base)
source "$conda_base"/etc/profile.d/conda.sh
conda activate "$envname"

# If we installed a conda environment, the this will just install ginput in develop mode so that it can be
# referenced whenever that environment is active.
pip install -e .

# After installation, create a version of run_ginput set up to use the environment
# Have the makefile call this with a default environment name, allow that to be overridden as a setting
./install-runscript.sh
