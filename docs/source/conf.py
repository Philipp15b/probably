# pylint: disable-all
import inspect
import os
import sys
from datetime import datetime

import sphinx_bootstrap_theme

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('.'))

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints', 'sphinx.ext.autosummary', 'sphinx.ext.doctest',
    'sphinx_click.ext', 'sphinx_git', 'sphinxcontrib.katex', 'exec_directive',
    'sphinx.ext.todo'
]

# General information about the project.
project = 'Probably'
copyright = '{}, Philipp Schröer'.format(datetime.now().year)
author = 'Philipp Schröer'
master_doc = 'index'
pygments_style = 'sphinx'
todo_include_todos = True

html_theme_path = ["themes"] + sphinx_bootstrap_theme.get_html_theme_path()
html_theme = 'fixedbootstrap'
_navbar_links = [
    ('GitHub Repository', 'https://github.com/Philipp15b/probably', True)
]
html_theme_options = {
    'navbar_sidebarrel': False,
    'globaltoc_depth': -1,
    'navbar_site_name': "Contents",
    'source_link_position': "footer",
    'bootswatch_theme': "united",
    'navbar_pagenav': False,
    'navbar_links': _navbar_links
}
html_static_path = ['_static']

template_path = ['_templates']
html_sidebars = {'**': ['simpletoctree.html']}
html_extra_path = [".nojekyll"]

# Do not show Python objects in the TOC tree, we have our own headings for
# everything. Not because we wanted to, but because this is a new feature in
# Sphinx that we didn't have when most of the documentation was written.
toc_object_entries = False

autodoc_default_options = {'members': True, 'undoc-members': True}
autodoc_member_order = "bysource"

#napoleon_use_ivar = True

# et typing.TYPE_CHECKING to True to enable “expensive” typing imports
set_type_checking_flag = False  # TODO
always_document_param_types = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'graphviz': ('https://graphviz.readthedocs.io/en/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None)
}

# nitpicky emits warnings for all broken links
# nitpicky = True

autodoc_type_aliases = {"Expr" : "probably.pgcl.ast.Expr", "Instr" : "probably.pgcl.ast.Instr", "Type" : "probably.pgcl.ast.Type",
                        "Decl" : "probably.pgcl.ast.Decl", "DistrExpr" : "probably.pgcl.ast.DistrExpr", "Query" : "probably.pgcl.ast.Query"}
