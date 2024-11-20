# Ginput: GGG meteorology and a priori VMR preprocessor

[![Documentation status](https://readthedocs.org/projects/ginput/badge/?version=latest)](https://ginput.readthedocs.io/en/latest/?badge=latest)

## Copyright notice

Copyright (c) 2022, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
* Neither the name of Caltech nor its operating division, the Jet Propulsion Laboratory, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Quickstart

* Make sure you have a Python installation with the `conda` package manager
  on your system. We currently recommend [miniforge](https://github.com/conda-forge/miniforge).
* Run `make install`
* Run `./run_ginput.py --help` to see available subcommands
* For more detailed help, try `man ginput` after running `make install`. 
  (If that doesn't work, try `man man/build/man/ginput.1` from this 
  directory or add `man/build/man/` to your `MANPATH`).
  
For more install options, run `make help`. 

## Is it working correctly?

To check whether your installation is working, there are example .mod, .vmr, .map, 
and .map.nc files in `ginput/testing/test_input_data` that have been generated from
both GEOS-FPIT and GEOS-FP. While TCCON uses GEOS-FPIT, it requires a data subscription,
so you may prefer to use GEOS-FP. 

To verify you have installed and are using `ginput` correctly, we recommend you generate
at least the .mod and .vmr files for Lamont (site code "oc") on 1 Jan 2018 and compare
against the pregenerated test files. Differences should be less than ~1%. 

If you have GEOS FP-IT files available on your system, the script 
`ginput/testing/test_input_data/geosfp-it/link_geos_fpit_files.sh` will help link them 
to the correct location in this repo to run the automatic test case. `cd` to 
`ginput/testing/test_input_data/geosfp-it/` and run `./link_geos_fpit_files.sh --help`
to see its usage. This requires the following files:

* The 8 model level assimilated quantity files for 2018-01-01 (`GEOS.fpit.asm.inst3_3d_asm_Nv.GEOS5124.20180101*.nc4`)
* The 8 2D assimilated quantity files for 2018-01-01 (`GEOS.fpit.asm.inst3_2d_asm_Nx.GEOS5124.20180101*.nc4`)
* The 8 model level chemistry files for 2018-01-01 (`GEOS.fpit.asm.inst3_3d_chm_Nv.GEOS5124.20180101*.nc4`)

To run the test, the easiest way is to activate the conda environment that `ginput` was installed
in (`ginput-auto-default` if installed with `make install`) and run `make test`. This will take
several minutes to run. If tests fail because the output .mod/.vmr/.map files do not match the expected,
plots will be generated in `ginput/testing/plots` for you to inspect the differences and verify that
they are reasonable.

## Documentation

Man pages are created in `man/build/man/` during installation and can be accessed with e.g. `man man/build/man/ginput.1`.
These may also be added to your MANPATH so that e.g. `man ginput` works, but this is system dependent. Full documentation
is provided at https://ginput.readthedocs.io/en/latest/.

Some additional documentation is provided on the TCCON wiki:

* [Running ginput](https://tccon-wiki.caltech.edu/Main/UsingGinput)
* [Code structure](https://tccon-wiki.caltech.edu/Main/GinputCodeStructure)

These will be folded into the readthedocs instance in the future.

## Terms of use

Ginput is licensed under the Apache license as of 8 Sept 2022. 
Prior to this date, it was licensed under the LGPL license.
If you download this software after 8 Sept 2022, you agree to abide by the terms of the
Apache license.
The full legal terms are contained in LICENSE.txt file. For a short summary, please see
[here](https://choosealicense.com/licenses/apache-2.0/#). If you have any questions about
use, please contact us (contact information is below).

In addition to the Apache license, you should cite the ginput paper in any publications
resulting from the use of ginput. (At time of release, the manuscript is still in preparation,
so contact us for the citation.) Please also consider contacting us to let us know you are
using ginput!

## Python support

Only Python 3 is supported. Python 2 reached end-of-life on 1 Jan 2020. 
We also require that the `conda` package manager provided with the Anaconda
Python distribution be installed, and so only officially support Anaconda or
Miniconda Python. 

If you have Anaconda or Miniconda based on Python 2 installed, that should work,
as ginput is configured to create a Python 3 environment for itself on install.
This also ensures that it's dependencies do not conflict with your existing 
setup.

## Contact

For assistance with `ginput`, contact [Josh Laughner](https://science.jpl.nasa.gov/people/joshua-laughner/).
