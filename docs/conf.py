"""Sphinx configuration for the track_gen documentation site."""
import importlib.metadata
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "_ext"))

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
    # Local checks (docs/_ext/trackgen_docscheck.py): every public symbol documented, no
    # roadmap / version jargon outside upgrading/*, every data file present.
    "trackgen_docscheck",
]

# Internal planning pages, review notes and the build tooling stay in the repository but
# out of the published site (nothing is moved or deleted).
exclude_patterns = [
    "_build",
    "_ext/**",
    "_tools/**",
    "_data/**",
    "superpowers/**",
    "library-readiness/**",
    "code-review-*.md",
    "Thumbs.db",
    ".DS_Store",
]
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

# Nitpicky: every cross-reference must resolve. Third-party targets that have no
# inventory are ignored by pattern; the remaining docstring-origin targets that do not
# resolve yet are listed in _tools/nitpick_allowlist.txt (generated from a nitpicky build;
# entries may only be removed as their targets are resolved). A new unresolved reference in
# a page fails the build. The allowlist is applied by trackgen_docscheck (not
# nitpick_ignore), which also fails a full build on an entry that no longer silences
# anything.
nitpicky = True
nitpick_ignore_regex = [
    # Warp ships no intersphinx inventory: wp.array, warp._src.types.array, wp.Stream, ...
    (r"py:.*", r"(warp|wp)(\..*)?"),
    # numpy aliases in signatures / docstrings that the numpy inventory does not hold
    (r"py:.*", r"(numpy|np)\..*"),
    # torch is optional and has no inventory configured
    (r"py:.*", r"torch(\..*)?"),
    # bare Warp type names written in docstring type fields
    (r"py:.*", r"vec[234][fi]?|quatf|mat[234][234]f?|u?int(8|16|32|64)|float(16|32|64)"
               r"|Stream|Graph|Device|array"),
]


def _read_allowlist(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            typ, target = line.split(None, 1)
            out.append((typ, target))
    return out


trackgen_nitpick_allowlist = _read_allowlist(_HERE / "_tools" / "nitpick_allowlist.txt")

# trackgen_docscheck: "error" makes every api / jargon finding a build warning, so
# `sphinx -W` fails on an undocumented public name or on roadmap / version jargon outside
# upgrading/*; "report" (local triage only) logs them as docscheck: ... lines and in
# <doctreedir>/docscheck.txt without failing. Missing data files always warn.
trackgen_docscheck = "error"
trackgen_jargon_exclude = ("upgrading/",)
# Public names that are deliberately not documented, with the reason.
trackgen_api_exempt = {
    # The library's own constructors call it to refuse construction inside a capture
    # (CaptureError); users meet the error, documented with track_gen.exec, not the helper.
    "track_gen.exec.runner.ensure_not_capturing":
        "internal helper of the library's constructors, not re-exported",
    # The stages catalogue: TrackGen's own built-in stage registry, reached through
    # track_gen.plugin.registry; not a plugin-facing API.
    "track_gen.stages.register": "built-in stage catalogue (internal)",
    "track_gen.stages.register_builtins": "built-in stage catalogue (internal)",
    "track_gen.stages.get": "built-in stage catalogue (internal)",
    "track_gen.stages.available": "built-in stage catalogue (internal)",
    "track_gen.stages.scoped": "built-in stage catalogue (internal)",
    "track_gen.stages.CHANNELS": "built-in channel catalogue (internal)",
    "track_gen.stages.Channel": "built-in channel catalogue (internal)",
    "track_gen.stages.FAIL": "built-in channel catalogue (internal)",
    "track_gen.stages.SEED": "built-in channel catalogue (internal)",
    "track_gen.stages.OPEN_KINDS": "built-in stage catalogue (internal)",
    "track_gen.stages.PARAM_WRITER_KINDS": "built-in stage catalogue (internal)",
    # The catalogue constants of the built-in stage modules (channel and field tables the
    # planner and the commit read).
    "track_gen.stages.channels.CARRIED_CATALOGUE": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.CHANNELS": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.Channel": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.DOMAINS": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.ELEVATION_GATE_CHANNELS": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.FAIL": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.INTERPS": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.PERS": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.PLUGIN_CHANNEL": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.SEED": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.channel_domain": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.family_key": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.is_plugin_channel": "built-in stage catalogue (internal)",
    "track_gen.stages.channels.required_port_domain": "built-in stage catalogue (internal)",
    "track_gen.stages.commit.ENV_FIELDS": "built-in stage catalogue (internal)",
    "track_gen.stages.commit.GATE_ENV_FIELDS": "built-in stage catalogue (internal)",
    "track_gen.stages.commit.GATE_POINT_FIELDS": "built-in stage catalogue (internal)",
    "track_gen.stages.commit.POINT_FIELDS": "built-in stage catalogue (internal)",
    "track_gen.stages.gates.CATALOGUE_STAGES": "built-in stage catalogue (internal)",
    "track_gen.stages.gates.GATE_EXPORTS": "built-in stage catalogue (internal)",
    "track_gen.stages.gates.REGISTRY_STAGES": "built-in stage catalogue (internal)",
    "track_gen.stages.track.BUILTIN_STAGES": "built-in stage catalogue (internal)",
    "track_gen.stages.track.CATALOGUE_STAGES": "built-in stage catalogue (internal)",
    "track_gen.stages.track.REGISTRY_STAGES": "built-in stage catalogue (internal)",
    "track_gen.stages.track.TRACK_EXPORTS": "built-in stage catalogue (internal)",
}

html_theme = "furo"
html_static_path = ["_static"]
html_title = "TrackGen"

linkcheck_timeout = 30
linkcheck_retries = 2
linkcheck_workers = 4
linkcheck_rate_limit_timeout = 120
# GitHub renders anchors client-side (README headings, line ranges): not checkable.
linkcheck_anchors_ignore_for_url = [r"https://github\.com/.*"]
# doi.org resolves to the publisher; those redirects are expected (IJEEI serves its
# papers over plain http).
linkcheck_allowed_redirects = {
    r"https://doi\.org/.*": r"https://(ieeexplore\.ieee\.org|linkinghub\.elsevier\.com"
                            r"|www\.sciencedirect\.com|saemobilus\.sae\.org"
                            r"|link\.springer\.com)/.*|https?://ijeei\.org/.*",
}
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
    # CMU's server drops bot connections (connection reset)
    r"https?://([a-z0-9-]+\.)*cmu\.edu/.*",
    # IEEE Xplore and Elsevier's linking hub refuse automated clients
    r"https://ieeexplore\.ieee\.org/.*",
    r"https://linkinghub\.elsevier\.com/.*",
]
