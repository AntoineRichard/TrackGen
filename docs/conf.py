"""Sphinx configuration for the track_gen documentation site."""
import importlib.metadata

project = "TrackGen"
author = "Antoine Richard"
copyright = "2026, Antoine Richard"
release = importlib.metadata.version("track_gen")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "superpowers/**", "Thumbs.db", ".DS_Store"]
add_module_names = False

# Autodoc / napoleon
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = True
napoleon_use_param = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_static_path = ["_static"]
html_title = "TrackGen"

linkcheck_ignore = [
    # Paywall / bot-blocked publisher sites (403 Forbidden)
    r"https://www\.tandfonline\.com/.*",
    r"https://asmedigitalcollection\.asme\.org/.*",
    r"https://dl\.acm\.org/.*",
    r"https://sourceforge\.net/.*",
    r"https://gitlab\.com/speed-dreams/.*",
    r"https://services\.igi-global\.com/.*",
    # DOI redirects that resolve to paywalled content
    r"https://doi\.org/10\.1080/.*",
    r"https://doi\.org/10\.1115/.*",
    r"https://doi\.org/10\.1145/.*",
    r"https://doi\.org/10\.4018/.*",
]
