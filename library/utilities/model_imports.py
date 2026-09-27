"""Load legacy deployed classifiers without importing unused sampler code.

Older exports refer to ndsampler.CategoryTree. Its tensor methods are
preserved in viame.classifiers.category_tree; importing the sampler itself also
imports its demo image backend.
Only exports using that class exclusively are rewritten. Other models use
ubelt's original loader unchanged.
"""
import ast
import hashlib
import importlib.abc
import importlib.util
import linecache
import sys


def category_tree_imports(source):
    """Return source with provably CategoryTree-only sampler imports replaced."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    edits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            return source  # Leave package-relative import handling to ubelt.
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != 'ndsampler':
                    continue
                binding = alias.asname or alias.name
                uses = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == binding]
                if not uses or any(not isinstance(parents[n], ast.Attribute) or
                                   parents[n].value is not n or
                                   parents[n].attr != 'CategoryTree' or
                                   not isinstance(n.ctx, ast.Load) for n in uses):
                    continue
                if any(isinstance(n, ast.arg) and n.arg == binding for n in ast.walk(tree)):
                    continue
                alias.name, alias.asname = 'viame.classifiers.category_tree', binding
                edits.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == 'ndsampler':
            if all(alias.name == 'CategoryTree' for alias in node.names):
                node.module = 'viame.classifiers.category_tree'
                edits.append(node)
    lines = source.splitlines(keepends=True)
    for node in sorted(set(edits), key=lambda n: (n.lineno,n.col_offset), reverse=True):
        # Keep line numbering for tracebacks and source inspection.
        if node.lineno == node.end_lineno:
            line = lines[node.lineno-1].encode()
            lines[node.lineno-1] = (line[:node.col_offset] + ast.unparse(node).encode() + line[node.end_col_offset:]).decode()
    return ''.join(lines)


def load_model_module(path, read_source):
    """Use the original path/zip loader unless CategoryTree imports changed."""
    import ubelt as ub
    source = read_source(path)
    rewritten = category_tree_imports(source)
    if rewritten == source:
        return ub.import_module_from_path(path)
    path = str(path)
    name = '_viame_deployed_' + hashlib.sha256((path+'\0'+rewritten).encode()).hexdigest()
    if name in sys.modules:
        return sys.modules[name]

    class Loader(importlib.abc.SourceLoader):
        def get_filename(self, fullname):
            return path
        def get_data(self, filename):
            return rewritten.encode()
        def get_code(self, fullname):
            return compile(rewritten, path, 'exec')

    loader = Loader()
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    linecache.cache[path] = (len(rewritten),None,rewritten.splitlines(True),path)
    try:
        loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module
