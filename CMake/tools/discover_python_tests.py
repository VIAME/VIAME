#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMake helper script to discover python tests

Usage:
    python discover_python_tests.py [fpath1, [fpaths ...]]
"""
from __future__ import print_function
import ast
import sys


class TopLevelVisitor(ast.NodeVisitor):
    """
    Parses top-level function and method names

    References:
        # For other visit_<classname> values see
        http://greentreesnakes.readthedocs.io/en/latest/nodes.html

    Example:
        >>> import sys
        >>> sys.path.append('/home/joncrall/code/VIAME/packages/kwiver/CMake/tools')
        >>> from discover_tests import *
        >>> import ubelt as ub
        >>> source = ub.codeblock(
            '''
            def test_foobar():
                def test_subfunc():
                    pass
            class TestSpam(object):
                def test_eggs():
                    pass
            ''')
        >>> self = TopLevelVisitor.parse(source)
        >>> assert 'test_foobar' in self.callnames
        >>> assert 'test_subfunc' not in self.callnames
        >>> assert 'TestSpam.test_eggs' in self.callnames
    """
    def __init__(self):
        super(TopLevelVisitor, self).__init__()
        self.testables = []
        self._current_classname = None

    @classmethod
    def parse(TopLevelVisitor, source):
        pt = ast.parse(source.encode('utf-8'))
        self = TopLevelVisitor()
        self.visit(pt)
        return self

    def visit_FunctionDef(self, node):
        if self._current_classname is None:
            callname = node.name
        else:
            callname = self._current_classname + '::' + node.name
        if node.name.startswith('test_'):
            self.testables.append(callname)

    def visit_ClassDef(self, node):
        callname = node.name
        self._current_classname = callname
        if callname.startswith('Test'):
            self.generic_visit(node)
        self._current_classname = None

    def visit_Assign(self, node):
        ast.NodeVisitor.generic_visit(self, node)


def discover_tests(fpath):
    with open(fpath, 'rb') as file_:
        source = file_.read().decode('utf-8')
    try:
        self = TopLevelVisitor.parse(source)
        return self.testables
    except Exception:  # nocover
        if fpath:
            print('Failed to parse tests for fpath=%r' % (fpath,))
        else:
            print('Failed to parse tests')
        raise


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/VIAME/packages/kwiver/CMake/tools
        python ~/code/VIAME/packages/kwiver/CMake/tools/discover_python_tests.py
        python discover_tests.py test_*.py
        import sys
    """
    module_fpaths = sys.argv[1:]
    for fpath in module_fpaths:
        testables = discover_tests(fpath)
        print('\n'.join(testables))
