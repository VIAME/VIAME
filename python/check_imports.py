import ast
import argparse
import sys
from importlib import util

class ImportVisitor(ast.NodeVisitor):
    def __init__(self, node, imports = set() ):
        self.ast_node = node
        self.imports = imports
    def visit(self, node=None):
        ast.NodeVisitor.visit(self, node)
    def generic_visit(self, node):
        ast.NodeVisitor.generic_visit(self, node)
    def visit_Import(self, node):
        ast.NodeVisitor.generic_visit(self, node)
    def visit_ImportFrom(self, node):
        self.imports.add(node.module)
    def visit_alias(self, node):
        self.imports.add(node.name)

def extract_top_level_modules(imports, skiplist = None):
    req_set = set()
    for import_ in imports:
        req = import_.split('.')[0]
        if skiplist and req not in skiplist:
            req_set.add(req)
    return req_set

def is_pkg_present(pkg):
    pkg_spec = util.find_spec(pkg)
    return True if pkg_spec else False

def generate_ast(filename, skiplist = None):
    file = open(filename)
    top_level_node = ast.parse(file.read())
    imports = set()
    ImportVisitor(top_level_node, imports).visit(top_level_node)
    reqs = extract_top_level_modules(imports, skiplist)
    for req in reqs:
        if not is_pkg_present(req):
            sys.exit(1)

def main():
    args = argparse.ArgumentParser(usage='Determine if python file dependencies are met by packages present in system')
    args.add_argument('--filename', '-f', action="store", required=True, dest="filename")
    args.add_argument('--ignore-imports', '-ii', action="store",nargs="+", required=False, dest="skiplist")
    parsed_args = args.parse_args(sys.argv[1:])
    generate_ast(parsed_args.filename, parsed_args.skiplist)

if __name__ =='__main__':
    main()
