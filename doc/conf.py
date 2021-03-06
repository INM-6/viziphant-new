# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import date

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Viziphant'
authors = 'Elephant team'
copyright = u"2019-{this_year}, {authors}".format(this_year=date.today().year,
                                                  authors=authors)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    'sphinx_tabs.tabs',
    'sphinx.builders.linkcheck',
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'


# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
root_dir = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(root_dir, 'viziphant', 'VERSION')) as version_file:
    # The full version, including alpha/beta/rc tags.
    release = version_file.read().strip()

# The short X.Y version.
version = '.'.join(release.split('.')[:-1])


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Required to automatically create a summary page for each function listed in
# the autosummary fields of each module.
autosummary_generate = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# TODO: create logo and favicon
# html_logo = 'images/viziphant_logo_sidebar.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = 'images/viziphant_favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Suppresses  wrong numpy doc warnings
# see here https://github.com/phn/pytpm/issues/3#issuecomment-12133978
numpydoc_show_class_members = False


# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# If false, no index is generated.
html_use_index = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# Output file base name for HTML help builder.
htmlhelp_basename = 'viziphantdoc'


# -- Options for LaTeX output --------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    # 'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual])
latex_documents = [
    ('index', 'elephant.tex', u'Elephant Documentation',
     authors, 'manual'),
]


# -- Options for manual page output --------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'elephant', u'Elephant Documentation',
     [authors], 1)
]


# -- Options for Texinfo output ------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index',
     'Elephant',
     u'Elephant Documentation',
     authors,
     'Elephant',
     'Elephant is a package for the analysis of neurophysiology data.',
     'Miscellaneous'),
]


# -- Options for Epub output ---------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = authors
epub_publisher = authors
epub_copyright = copyright


# configuration for intersphinx: refer to Elephant
# intersphinx_mapping = {'elephant': ('https://elephant.readthedocs.io', None)}

# Use more reliable mathjax source
mathjax_path = 'https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
