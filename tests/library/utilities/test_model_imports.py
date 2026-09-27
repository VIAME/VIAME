"""Legacy CategoryTree imports should not load the sampler's image demos."""
from pathlib import Path
import pytest
from viame.utilities.model_imports import category_tree_imports, load_model_module

@pytest.mark.parametrize('source', [
    'import ndsampler\nx = ndsampler.CategoryTree\n',
    'import ndsampler as ns\nx = ns.CategoryTree\n',
    'from ndsampler import CategoryTree\nx = CategoryTree\n',
    'def build():\n    import ndsampler\n    return ndsampler.CategoryTree\n',
])
def test_category_tree_only_imports(source):
    rewritten = category_tree_imports(source)
    assert 'viame.classifiers.category_tree' in rewritten
    namespace = {}
    exec(rewritten, namespace)
    from viame.classifiers import category_tree
    result = namespace['build']() if 'build' in namespace else namespace['x']
    assert result is category_tree.CategoryTree

@pytest.mark.parametrize('source', [
    'import ndsampler\nx = ndsampler.CocoSampler\n',
    'import ndsampler\nx = getattr(ndsampler,"CategoryTree")\n',
    'import ndsampler\ndef build(ndsampler):\n    return ndsampler.CategoryTree\n',
    'from .helpers import x\nimport ndsampler\ny = ndsampler.CategoryTree\n',
])
def test_other_imports_preserve_original_source(source):
    assert category_tree_imports(source) == source


def test_model_loader_preserves_origin_and_source_inspection(tmp_path):
    import inspect
    path = tmp_path/'model.py'
    source = 'def build():\n    import ndsampler\n    return ndsampler.CategoryTree\n'
    path.write_text(source)
    module = load_model_module(str(path), lambda p: Path(p).read_text())
    from viame.classifiers import category_tree
    assert module.build() is category_tree.CategoryTree
    assert module.__file__ == str(path)
    assert 'viame.classifiers.category_tree' in inspect.getsource(module.build)
    assert path.read_text() == source
    assert load_model_module(str(path), lambda p: Path(p).read_text()) is module
