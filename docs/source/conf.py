'''
Date: 2024-11-28 17:46:44
LastEditors: muzhancun muzhancun@126.com
LastEditTime: 2025-01-16 15:01:47
FilePath: /MineStudio/docs/source/conf.py
'''
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from sphinx.application import Sphinx
from sphinx.locale import _
import pydata_sphinx_theme
sys.path.append(str(Path(".").resolve()))

from custom_directives import generate_versions_json

project = 'MineStudio'
copyright = str(datetime.now().year) + ", The CraftJarvis Team"
author = 'The CraftJarvis Team'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.graphviz",
    "sphinxext.rediraffe",
    "sphinx_design",
    "sphinx_copybutton",
    # "autoapi.extension",
    # For extension examples and demos
    "myst_parser",
    "ablog",
    "jupyter_sphinx",
    "sphinxcontrib.youtube",
    "nbsphinx",
    "numpydoc",
    "sphinx_togglebutton",
    # "jupyterlite_sphinx",
    "sphinx_favicon",
    "sphinx_multiversion",
]

templates_path = ['_templates']
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

sys.path.insert(0, os.path.abspath('../minestudio'))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
# htmp_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = [
    "custom.css"
]

html_theme_options = {
  "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/CraftJarvis/MineStudio",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/minestudio",
            "icon": "fa-custom fa-pypi",
        },
        {
            "name": "ArXiv",
            "url": "https://arxiv.org/pdf/2310.08235",
            "icon": "ai ai-arxiv",
        }
  ], 
  "navbar_align": "left",
  "show_toc_level": 1,
  "navbar_center": ["version-switcher", "navbar-nav"],  
  "logo": {
    "text": "MineStudio",
    "image_light": "_static/logo-no-text-light.svg", 
    "image_dark": "_static/logo-no-text-light.svg",
  },
  "navbar_start": ["navbar-logo"],  # 在导航栏显示 Logo
  "switcher": {
        "json_url": "https://craftjarvis.github.io/MineStudio/_static/switcher.json",
        "version_match": "master",
   }
}

html_title = f"MineStudio {release}"
html_favicon = "_static/logo-no-text-light.svg"

# -- application setup -------------------------------------------------------


def setup_to_main(
    app: Sphinx, pagename: str, templatename: str, context, doctree
) -> None:
    """
    Add a function that jinja can access for returning an "edit this page" link
    pointing to `main`.
    """

    def to_main(link: str) -> str:
        """
        Transform "edit on github" links and make sure they always point to the
        main branch.

        Args:
            link: the link to the github edit interface

        Returns:
            the link to the tip of the main branch for the same file
        """
        links = link.split("/")
        idx = links.index("edit")
        return "/".join(links[: idx + 1]) + "/main/" + "/".join(links[idx + 2 :])

    context["to_main"] = to_main


def setup(app: Sphinx) -> Dict[str, Any]:
    """Add custom configuration to sphinx app.

    Args:
        app: the Sphinx application
    Returns:
        the 2 parallel parameters set to ``True``.
    """

    generate_versions_json()

    app.connect("html-page-context", setup_to_main)

    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

# setup multiversion
smv_branch_whitelist = r'^(master|releases/.*)$' # only include master and releases/* branches