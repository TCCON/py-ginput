# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
# Include the directory with ginput in it so autodoc finds the modules
sys.path.insert(0, os.path.abspath('../..'))
from ginput import __version__ as release # noqa
# Extract the version from ginput itself
print(f'Using release = {release}')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ginput'
copyright = '2023, All rights reserved'
author = 'Joshua Laughner and Sebastien Roche'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme', 'myst_parser', 'sphinx.ext.napoleon', 'sphinx.ext.inheritance_diagram']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
