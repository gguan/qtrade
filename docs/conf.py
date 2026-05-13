from __future__ import annotations

import os
import sys
import time

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QTrade'
copyright = f'{time.localtime().tm_year}, Guan Guan'
author = 'Guan Guan'
release = '0.5.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',    
    'myst_parser',
    'sphinx_copybutton',
    'sphinx.ext.githubpages',
    'sphinx.ext.coverage',
    # 'sphinx_gallery.gen_gallery',
    'sphinx_github_changelog',
]

templates_path = ['_templates']
exclude_patterns = ['_build', '404.md', 'locale']

# -- Internationalization ----------------------------------------------------
# Build the docs in either English (default) or Chinese with
# ``-D language=zh_CN``. Untranslated strings fall back to the English
# source, so the Chinese build starts useful from day one and improves
# as ``locale/zh_CN/LC_MESSAGES/*.po`` files are filled in.
language = 'en'
locale_dirs = ['locale/']
gettext_compact = False  # one .pot template per source file
gettext_uuid = True      # stable msgid hashes across regenerations

# Napoleon settings
napoleon_use_ivar = True
napoleon_use_admonition_for_references = True
napoleon_custom_sections = [("Returns", "params_style")]

# Autodoc
autoclass_content = "both"
autodoc_preserve_defaults = True

# This function removes the content before the parameters in the __init__ function.
# This content is often not useful for the website documentation as it replicates
# the class docstring.
def remove_lines_before_parameters(app, what, name, obj, options, lines):
    if what == "class":
        # ":param" represents args values
        first_idx_to_keep = next(
            (i for i, line in enumerate(lines) if line.startswith(":param")), 0
        )
        lines[:] = lines[first_idx_to_keep:]


def setup(app):
    app.connect("autodoc-process-docstring", remove_lines_before_parameters)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = "QTrade"
html_static_path = ['_static']
html_js_files = ['lang_switcher.js']

# Inject a language switcher into the bottom of furo's left sidebar
# (after the toctree, before scroll-end). The JS in lang_switcher.js
# rewrites the link's href / label based on the URL ("/en/" ↔ "/zh/").
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/lang_switcher.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
    ]
}
