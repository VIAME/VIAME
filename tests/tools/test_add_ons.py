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


def test_machine_progress_covers_download_and_extraction(tmp_path, monkeypatch, capsys):
    import io
    import json
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('configs/pipelines/models/fish.pt', b'x' * (2 << 20))
    content = archive.read_bytes()
    response = io.BytesIO(content)
    response.headers = {'Content-Length': str(len(content))}
    monkeypatch.setenv('VIAME_ADDON_PROGRESS', '1')
    monkeypatch.setattr(m.urllib.request, 'urlopen', lambda *args, **kwargs: response)
    addon = m.Addon('fish', 'https://example.test/fish.zip', '', '', '', [], '')
    m.install_addon(tmp_path, addon)
    events = [json.loads(line.split(' ', 1)[1]) for line in capsys.readouterr().out.splitlines()
              if line.startswith('VIAME_ADDON_PROGRESS ')]
    assert [event['phase'] for event in events][0] == 'download'
    assert any(event['phase'] == 'verify' for event in events)
    assert any(event['phase'] == 'download' and event['done'] == len(content)
               and event['total'] == len(content) for event in events)
    assert events[-1] == {'phase': 'install', 'done': 100, 'total': 100}
    assert (tmp_path / 'configs/pipelines/models/fish.pt').stat().st_size == 2 << 20


def test_incomplete_download_is_an_error(tmp_path, monkeypatch):
    import io
    response = io.BytesIO(b'too short')
    response.headers = {'Content-Length': '1000'}
    monkeypatch.setattr(m.urllib.request, 'urlopen', lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match='Incomplete download'):
        m.download('https://example.test/file.zip', tmp_path / 'file.zip')
