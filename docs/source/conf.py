import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GENetLib"
author = "Yuhao Zhong"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "sphinx_gallery.load_style",
    "sphinx.ext.viewcode",
    "nbsphinx_link",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
]

napoleon_google_docstring = False

templates_path = ["_templates"]
exclude_patterns = []

autodoc_inherit_docstrings = True
autodoc_preserve_defaults = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": True,
}
autoclass_content = "both"

nb_execution_mode = "auto"
nbsphinx_allow_errors = False
nbsphinx_custom_formats = {
    ".pct.py": ["jupytext.reads", {"fmt": "py:percent"}],
}
nbsphinx_execute_arguments = ["--InlineBackend.figure_formats={'svg', 'pdf'}"]
# If window is narrower than this, input/output prompts are on separate lines:
nbsphinx_responsive_width = "700px"
# List expensive-to-compute notebooks here:
# nb_execution_excludepatterns = ['list', 'of', '*patterns']
# Alias kernel names
nb_kernel_rgx_aliases = {"GENetLib": "python3"}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "GENetLib's documentation"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_sidebars = {"**": ["sidebar-nav-bs"]}
html_theme_options = {
    "primary_sidebar_end": [],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Barry57/GENetLib",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": False,
    "collapse_navigation": True,
    "logo": {
        "image_light": "logo.png",
        "image_dark": "logo.png",
        "text": html_title,
        "alt_text": "GENetLib's logo",
    },
}
html_context = {
    "github_user": "Barry57",
    "github_repo": "GENetLib",
    "github_version": "dev",
    "doc_path": "docs",
    "default_mode": "light",
}
htmlhelp_basename = "GENetLib's documentation"
html_show_sourcelink = False
