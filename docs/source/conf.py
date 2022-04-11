"""The sphinx config file."""
# -- Project information -----------------------------------------------------
from __future__ import annotations

import morphoclass

project = "Morphology-Classification"
copyright = "2020, Stanislav Schmidt"
author = "Stanislav Schmidt"
version = morphoclass.__version__

# -- Customization -----------------------------------------------------------
html_title = "MorphoClass"
html_show_sourcelink = False

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.coverage",
    # "sphinx.ext.doctest",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    # "canonical_url": "",
    # "analytics_id": "UA-XXXXXXX-1",  # Provided by Google in your dashboard
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "green",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": -1,
    "includehidden": True,
    "titles_only": False,
}
