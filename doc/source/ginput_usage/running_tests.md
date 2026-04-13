# Running tests

`ginput` now uses [pytest](https://docs.pytest.org/en/stable/) to run its tests.
Tests are broken down into three categories:

- **quick**: these are reasonably fast tests that check individual aspects of `ginput`'s behavior.
- **slow**: these are longer running tests, usually checking end-to-end behavior or a significant number of variations of a potential error.
- **glacial**: these are like _slow_ tests, but take even longer to run.

The simplest way to run tests is to:

1. activate the conda or virtual environment ginput is installed into, and
2. call `make quicktest`, `make test`, or `make fulltest` from the top directory.

If you want more control, you can run `pytest` directly.
This works best if you call it from the top directory and pass the `tests` subdirectory as the position argument to `pytest`, e.g.:

```
python -k "test_mod_files_jan2018" tests/
```

if you only wanted to run the `test_mod_files_jan2018` test.

(large-input-data)=
## Large input data

Some of the tests require very large input files that cannot be stored directly in the Git repository.
As part of test setup, if any test requiring these input files is to be run, it will check if those files
need downloaded.
This requires downloading one or more files from https://data.caltech.edu/, and can take a few minutes to complete.
If you want to skip that check, you can use the `GINPUT_TEST_SKIP_LARGE_FILE_CHECK`, described in the
[environmental variables](#environmental-variables) section, below.


By default, it will _not_ overwrite existing large files, and only print a warning if their checksum differs.
This is intended to avoid accidentally overwriting files being updated during development.
You can change this behavior with the `GINPUT_TEST_CLOBBER_LARGE_FILES`
[environmental variable](#environmental-variables).

## Tests directory structure

Under the `tests` directory, you will find the following subdirectories:

- `plots`: This is where plots automatically created by failing tests go. It also contains scripts to
  generate some plots useful to manually review the results of tests. If a test fails because the
  current values do not match the expected, check here to see if the test generated a plot first.
- `test_input_data`: This contains two types of files:
    1. data needed by ginput to run the tests, and
    2. expected results for tests that have benchmark data.
- `test_output_data`: This contains any files that are written by the tests. This can either be
  because we want to compare the contents of those files against a benchmark, or to generate
  updated benchmarks for future tests.

If you write new tests, set them up to take any required input from the `test_input_data` directory
and write any output files to the `test_output_data` directory.
If you want the test to compare against previous output, put the expected output under `test_input_data`
as well.

```{warning}
GitHub has fairly stringent [file size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Please take care not to commit large volumes of files to the repository. That means:

1. Avoid committing _any_ data files until you are sure they are correct. This means that
   you should not commit any new files under `test_input_data` until you have finished
   developing the feature and tests that use those data. This avoids bloating the repository
   with multiple copies of the data files.
   (Alternatively, [squash](https://stackoverflow.com/questions/5189560/how-do-i-squash-my-last-n-commits-together)
   the commits with multiple iterations of the data files before pushing to GitHub.)
2. Do not commit files that are more than 1-2 MB in size to the repo. This is why the
   [`test_input_data/large-files` directory](#large-input-data) and its associated CaltechData record exist.
   Try to reuse existing large files whenever possible.
   If you need to add new large files, [open an issue](https://github.com/TCCON/py-ginput/issues).
```

(environmental-variables)=
## Environmental variables

The following environmental variables affect the execution of certain tests:

- `GINPUT_TEST_SKIP_GEN`: Setting this to `1` will cause several of the very long running tests
  to skip generating the new output files and simply compare existing output files (under
  `test_output_data`) against the expected files. This is most useful when debugging the
  test itself; usually, you will want the output file generated each time to test the
  `ginput` code.
- `GINPUT_TEST_NPROCS`: This controls how many processes certain tests that have parallelized code
  use. The default is usually `8`, pass any integer >= 1 to modify this. Note that this does not
  use multiple threads to run tests in parallel; it only speeds up parts of `ginput` that are
  internally parallelized.
- `GINPUT_TEST_SKIP_LARGE_FILE_CHECK`: Setting this to `1` will skip trying to download the large
  files. Since the test fixture will otherwise at least verify the checksums of the large files,
  this can speed up testing when you know that you already have the correct files downloaded.
- `GINPUT_TEST_CLOBBER_LARGE_FILES`: By default, the test fixture will not overwrite large files
  that already exist locally with the remote copy. Setting this to `1` will cause it to overwrite
  local copies if they have a different checksum than expected.
