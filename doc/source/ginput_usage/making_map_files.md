(usage-map)=
# Making .map and .map.nc files

`.map` (*m*odel *a p*riori) files are a condensed file format that stores the a priori VMR profiles used by GGG.
These files aren't actually read by GGG, but are used to distribute the key a priori profiles to users who do
not want the full complexity of the `.vmr` or `.mav` files. Traditionally, these are generated during GGG post-processing
from the `.mav` file, but ginput can generate them directly.

## Creating .map or .map.nc files

If we assume we're continuing on from {ref}`usage-vmr`, and that we want text `.map` file to be written to `MAP_FILE_DIR`, then our command is this:

```
$ ./run_ginput.py map \
    --save-dir MAP_FILE_DIR \
    --product fp \
    --lat 36.604 \
    --lon -97.486 \
    --map-fmt txt \
    20180101 \
    MOD_FILE_DIR/fp/xx/vertical \
    VMR_FILE_DIR/fp/xx/vmrs-vertical
```

If you want netCDF `.map` files instead, change `--map-fmt txt` to `--map-fmt nc`.

As with the other sections, if you are generating `.map` files for different locations or dates, make sure to pass the correct `--lat`, `--lon` and date range values.

Unlike `.mod` and `.vmr` files, there is no option to generate `.map` files for a runlog.
This is because GGG does not require `.map` files as input (but it does need `.mod` and `.vmr` files), so there is little use for generating the `.map` files corresponding to the `.mod` and `.vmr` files required by a runlog.
