#!/usr/bin/env python

"""
This module is designed to used with _livereload to 
make it a little easier to write Sphinx documentation.
Simply run the command::
    python sphinx_server.py

and browse to http://localhost:5500

livereload_: https://pypi.python.org/pypi/livereload
"""

from livereload import Server, shell
server = Server()
server.watch('*.rst', shell('make html', cwd='.'))
server.watch('examples/*.rst', shell('make html', cwd='.'))
server.watch('conf.py', shell('make html', cwd='.'))
server.serve(root='_build/html')
