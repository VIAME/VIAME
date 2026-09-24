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


def test_seed_model_falls_back_only_when_allowed(fake_rfdetr, tmp_path, monkeypatch, capsys):
    namespace = {'os': os, 'parse_bool': lambda v: str(v).strip().lower() in ('1', 'true', 'yes', 'on')}
    load_function('rfdetr_default_pretrain_file', namespace)
    load_function('find_rfdetr_seed_weights', namespace)
    resolve = load_function('resolve_rfdetr_seed', namespace)
    large = type('RFDETRLarge', (), {})
    monkeypatch.delenv('VIAME_INSTALL', raising=False)

    present = tmp_path / 'my_seed.pth'
    present.write_bytes(b'weights')
    assert resolve(str(present), False, large) == str(present)

    missing = str(tmp_path / 'models' / 'rf-detr-large-2026.pth')
    with pytest.raises(ValueError, match='seed_model does not exist'):
        resolve(missing, False, large)
    # Allowed: the missing add-on copy falls through to the default weights,
    # which rfdetr downloads when no installed copy exists either.
    assert resolve(missing, 'True', large) == ''
    assert 'is not present' in capsys.readouterr().out

    # An installed copy elsewhere still wins over the download.
    installed = tmp_path / 'install' / 'configs' / 'pipelines' / 'models'
    installed.mkdir(parents=True)
    (installed / 'rf-detr-large-2026.pth').write_bytes(b'weights')
    monkeypatch.setenv('VIAME_INSTALL', str(tmp_path / 'install'))
    assert resolve(missing, True, large) == str(installed / 'rf-detr-large-2026.pth')
    assert resolve('', False, large) == str(installed / 'rf-detr-large-2026.pth')


def test_seed_model_url_fallback_fetches_the_file(fake_rfdetr, tmp_path, monkeypatch):
    namespace = {'os': os, 'parse_bool': lambda v: str(v).strip().lower() in ('1', 'true', 'yes', 'on')}
    load_function('rfdetr_default_pretrain_file', namespace)
    load_function('find_rfdetr_seed_weights', namespace)
    fetched = []

    def fake_download(url, seed_path):
        fetched.append(url)
        os.makedirs(os.path.dirname(seed_path), exist_ok=True)
        with open(seed_path, 'wb') as f:
            f.write(b'weights')
        return seed_path
    namespace['download_rfdetr_seed'] = fake_download
    resolve = load_function('resolve_rfdetr_seed', namespace)
    large = type('RFDETRLarge', (), {})
    missing = str(tmp_path / 'models' / 'rf-detr-large-2026.pth')
    assert resolve(missing, 'https://example.test/rf-detr-large-2026.pth', large) == missing
    assert fetched == ['https://example.test/rf-detr-large-2026.pth']
    # Present afterwards, so no second fetch.
    assert resolve(missing, 'https://example.test/rf-detr-large-2026.pth', large) == missing
    assert len(fetched) == 1


def test_download_rfdetr_seed_lands_beside_the_config_or_in_the_cache(tmp_path, monkeypatch):
    namespace = {'os': os}
    download = load_function('download_rfdetr_seed', namespace)
    served = tmp_path / 'served.pth'
    served.write_bytes(b'served weights')
    url = served.as_uri()
    target = tmp_path / 'models' / 'rf-detr-large-2026.pth'
    target.parent.mkdir()
    assert download(url, str(target)) == str(target)
    assert target.read_bytes() == b'served weights'
    # An unwritable install folder sends the file to rfdetr's cache instead.
    monkeypatch.setenv('RF_HOME', str(tmp_path / 'cache'))
    monkeypatch.setattr(os, 'access', lambda path, mode: False)
    assert download(url, str(tmp_path / 'readonly' / 'rf-detr-large-2026.pth')) == str(tmp_path / 'cache' / 'rf-detr-large-2026.pth')
