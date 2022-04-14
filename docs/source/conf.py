"""The sphinx config file."""
# -- Project information -----------------------------------------------------
from __future__ import annotations

import morphoclass

project = "Morphology-Classification"
copyright = "2022 Blue Brain Project, EPFL"
author = "Blue Brain Project, EPFL"
version = morphoclass.__version__
release = morphoclass.__version__

# -- Customization -----------------------------------------------------------
html_title = "MorphoClass"
html_show_sourcelink = False

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
# RTD theme options are described here:
# https://sphinx-rtd-theme.readthedocs.io/en/latest/configuring.html
html_theme_options = {
    # "analytics_id": "G-XXXXXXXXXX",  # Provided by Google in your dashboard
    "analytics_anonymize_ip": True,
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
