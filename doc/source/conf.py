# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import re

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ginput'
copyright = '2023, All rights reserved'
author = 'Joshua Laughner and Sebastien Roche'

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
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme', 'myst_parser', 'sphinx.ext.napoleon', 'sphinx.ext.inheritance_diagram']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
