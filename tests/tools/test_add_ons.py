import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_add_ons", Path(__file__).resolve().parents[2] / "tools" / "add_ons.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import zipfile


def test_failed_install_rolls_back(tmp_path):
    target = tmp_path / 'configs/pipelines'
    target.mkdir(parents=True)
    (target / 'a.pipe').write_text('old')
    archive = tmp_path / 'pack.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('configs/pipelines/a.pipe', 'new')
        z.writestr('configs/pipelines/b.pipe', 'second')
    replace = m.os.replace
    def fail_second(src, dst):
        if str(dst).endswith('b.pipe'):
            raise OSError('simulated failure')
        return replace(src, dst)
    with patch.object(m.os, 'replace', side_effect=fail_second), pytest.raises(OSError):
        m.install_archive(tmp_path, archive)
    assert (target / 'a.pipe').read_text() == 'old'
    assert not (target / 'b.pipe').exists()


@pytest.mark.parametrize('name', ['../escape', r'C:\escape', r'folder\..\escape'])
def test_unsafe_member_rejected(tmp_path, name):
    archive = tmp_path / 'pack.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(name, 'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        m.install_archive(tmp_path, archive)


def test_force_does_not_bypass_checksum(tmp_path):
    archive = tmp_path / 'pack.zip'
    archive.write_bytes(b'wrong')
    addon = m.Addon('test', '', '', 'expected', '', [], '')
    with pytest.raises(RuntimeError, match='checksum'):
        m.install_addon(tmp_path, addon, archive, force=True)


def test_failed_rollback_retains_backups(tmp_path):
    target = tmp_path / 'configs/pipelines'
    target.mkdir(parents=True)
    (target / 'a.pipe').write_text('old')
    archive = tmp_path / 'pack.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('configs/pipelines/a.pipe', 'new')
        z.writestr('configs/pipelines/b.pipe', 'second')
    replace = m.os.replace
    def fail(src, dst):
        if str(dst).endswith('b.pipe') or Path(src).name.startswith('backup-'):
            raise OSError('simulated filesystem failure')
        return replace(src, dst)
    with patch.object(m.os, 'replace', side_effect=fail), pytest.raises(RuntimeError, match='backups retained'):
        m.install_archive(tmp_path, archive)
    recovery, = tmp_path.glob('.viame-addon-*-recovery')
    assert (recovery / 'backup-0').read_text() == 'old'
    assert (recovery / 'recovery.json').exists()
