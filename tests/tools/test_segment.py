import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_segment", Path(__file__).resolve().parents[2] / "tools" / "segment.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_validation_preserves_all_non_polygon_fields(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    a.write_text('7,img.png,0,0,0,10,10,1,-1,fish,1\n')
    b.write_text('7,other.png,0,0,0,10,10,0.1,-1,shark,1\n')
    assert not m.validate_unit(a, b)['ok']


def test_manual_polygon_cannot_be_replaced(tmp_path):
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    row = '7,img.png,0,0,0,10,10,1,-1,fish,1,'
    a.write_text(row + '(poly) 0 0 10 0 0 10\n')
    b.write_text(row + '(poly) 0 0 5 0 0 5\n')
    assert not m.validate_unit(a, b)['ok']


def test_atomic_write_failure_keeps_original(tmp_path):
    p = tmp_path / 'a.csv'
    p.write_text('original')
    def broken():
        yield 'partial'
        raise OSError('disk failure')
    with pytest.raises(OSError):
        m.write_lines_atomic(p, broken())
    assert p.read_text() == 'original'


@pytest.mark.parametrize('layout', ['bin', 'Scripts', 'desktop-windows'])
def test_find_install_without_linux_setup(tmp_path, monkeypatch, layout):
    root = tmp_path / 'install'
    configs = root / 'configs'
    configs.mkdir(parents=True)
    script = configs / 'segment.py'
    script.touch()
    if layout == 'desktop-windows':
        (root / 'setup_viame.bat').touch()
    else:
        (root / layout).mkdir()
    monkeypatch.delenv('VIAME_INSTALL', raising=False)
    monkeypatch.setattr(m, '__file__', str(script))
    assert m.find_viame_install() == str(root)
    # A source checkout containing tools alone must not count as an install.
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'segment.py').touch()
    assert not m.is_viame_install(str(source))


@pytest.mark.parametrize('packaged', [True, False])
def test_sam2_import_matches_install_layout(packaged):
    import types
    builder, predictor = object(), object()
    calls = []
    def load(name):
        calls.append(name)
        if name == 'viame.sam2' and not packaged:
            raise ModuleNotFoundError("No module named 'viame.sam2'", name=name)
        return types.SimpleNamespace(build_sam2=builder, SAM2ImagePredictor=predictor)
    with patch('importlib.import_module', side_effect=load):
        assert m.sam2_classes() == (builder, predictor)
    prefix = 'viame.sam2' if packaged else 'sam2'
    assert calls[-2:] == [prefix + '.build_sam', prefix + '.sam2_image_predictor']


def test_sam2_dependency_errors_are_not_hidden():
    with patch('importlib.import_module', side_effect=ModuleNotFoundError(
            "No module named 'hydra'", name='hydra')):
        with pytest.raises(ModuleNotFoundError, match='hydra'):
            m.sam2_classes()
