# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'ginput'
copyright = '2021, Joshua Laughner, Sebastien Roche, and Matthaeus Kiel'
author = 'Joshua Laughner, Sebastien Roche, and Matthaeus Kiel'

# Extract the version from setup.py
release = 'x.y.z'
_setup_py_file = os.path.join(os.path.dirname(__file__), '..', '..', 'setup.py')
with open(_setup_py_file) as _f:
    for _line in _f:
        if 'version' in _line:
            # Assume the line in setup.py looks like version='1.2', version='1.2.0', or similar.
            # Basically there should always be a major and minor version, but the patch version
            # can be missing or be something more complex than a number.
            release = re.search(r"'(\d+\.\d+.*)'", _line).group(1)
print(f'Using release = {release}')


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('ginputman', 'ginput', 'Ginput: main documentation', [author], 1),
    ('getg5man', 'ginput.getg5', 'Ginput: download GEOS documentation', [author], 1),
    ('modman', 'ginput.mod', 'Ginput: mod file documentation', [author], 1),
    ('vmrman', 'ginput.vmr', 'Ginput: vmr file documentation', [author], 1),
    ('mapman', 'ginput.map', 'Ginput: map file documentation', [author], 1),
    ('aciman', 'ginput.acos', 'Ginput: ACOS interface documentation',
     [author], 1), 
    ('upnoaaman', 'ginput.upnoaa', 'Ginput: Update MLO/SMO monthly files', [author], 1)
]
