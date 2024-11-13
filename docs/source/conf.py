# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GENetLib's documentation"
copyright = '2024, Yuhao Zhong'
author = 'Yuhao Zhong'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_documatt_theme'
]

bibtex_bibfiles = ['references.bib']
bibtex_default_style = 'plain'

templates_path = ['_templates']
html_static_path = ["_static"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_documatt_theme"

html_logo = html_favicon = '_static/logo.png'

html_theme_options = {
    'header_text': 'GENetLib\'s documentation',
    'header_logo_style': 'width: 3rem;',
    "motto": "GENetLib is a Python library for gene–environment interaction analysis via deep learning."
}

html_sidebars = {
    'index': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
    '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html'],
}