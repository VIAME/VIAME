"""The RF-DETR trainer seeds from the add-on's installed COCO weights when
present, without importing rfdetr or torch."""
import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3] / 'plugins/pytorch'


def load_function(name, namespace):
    tree = ast.parse((ROOT / 'utilities.py').read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'utilities.py', 'exec'), namespace)
    return namespace[name]


def config_class(default):
    return SimpleNamespace(model_fields={'pretrain_weights': SimpleNamespace(default=default)})


@pytest.fixture
def fake_rfdetr(monkeypatch):
    """rfdetr.config as installed: RFDETRLarge builds its config on demand,
    so only the class named after the variant carries the default."""
    package = ModuleType('rfdetr')
    config = ModuleType('rfdetr.config')
    config.RFDETRLargeConfig = config_class('rf-detr-large-2026.pth')
    config.RFDETRNanoConfig = config_class('rf-detr-nano.pth')
    package.config = config
    monkeypatch.setitem(sys.modules, 'rfdetr', package)
    monkeypatch.setitem(sys.modules, 'rfdetr.config', config)
    return config


def test_default_file_comes_from_the_pinned_or_named_config(fake_rfdetr):
    namespace = {'os': os}
    default_file = load_function('rfdetr_default_pretrain_file', namespace)
    large = type('RFDETRLarge', (), {})
    nano = type('RFDETRNano', (), {'_model_config_class': config_class('rf-detr-nano.pth')})
    base = type('RFDETRBase', (), {'_model_config_class': config_class(None)})
    assert default_file(large) == 'rf-detr-large-2026.pth'
    assert default_file(nano) == 'rf-detr-nano.pth'
    assert default_file(base) is None


def test_installed_weights_are_found_in_the_config_dir_or_the_install(fake_rfdetr, tmp_path, monkeypatch):
    namespace = {'os': os}
    load_function('rfdetr_default_pretrain_file', namespace)
    find = load_function('find_rfdetr_seed_weights', namespace)
    large = type('RFDETRLarge', (), {})

    monkeypatch.delenv('VIAME_INSTALL', raising=False)
    assert find(large, ['', str(tmp_path / 'missing')]) is None

    install_models = tmp_path / 'install' / 'configs' / 'pipelines' / 'models'
    install_models.mkdir(parents=True)
    (install_models / 'rf-detr-large-2026.pth').write_bytes(b'weights')
    monkeypatch.setenv('VIAME_INSTALL', str(tmp_path / 'install'))
    assert find(large, ['']) == str(install_models / 'rf-detr-large-2026.pth')

    # An explicitly configured folder wins over the install.
    configured = tmp_path / 'configured'
    configured.mkdir()
    (configured / 'rf-detr-large-2026.pth').write_bytes(b'weights')
    assert find(large, [str(configured)]) == str(configured / 'rf-detr-large-2026.pth')

    # A variant without a default never claims a file.
    base = type('RFDETRBase', (), {'_model_config_class': config_class(None)})
    assert find(base, [str(configured)]) is None
