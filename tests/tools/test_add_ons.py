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


@pytest.mark.parametrize('cancel_after', [1, 2])
def test_cancel_restores_replaced_files(tmp_path, monkeypatch, cancel_after):
    target = tmp_path / 'configs/pipelines'
    target.mkdir(parents=True)
    (target / 'a.pipe').write_text('old')
    archive = tmp_path / 'pack.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('configs/pipelines/a.pipe', 'new')
        z.writestr('configs/pipelines/new/b.pipe', 'second')
    cancel = tmp_path / 'cancel'
    monkeypatch.setenv('VIAME_ADDON_CANCEL_FILE', str(cancel))
    replace = m.os.replace
    count = 0
    def request_cancel(src, dst):
        nonlocal count
        replace(src, dst)
        if Path(src).name.startswith('payload-'):
            count += 1
            if count == cancel_after:
                cancel.touch()
    monkeypatch.setattr(m.os, 'replace', request_cancel)
    with pytest.raises(m.InstallationCancelled):
        m.install_archive(tmp_path, archive)
    assert (target / 'a.pipe').read_text() == 'old'
    assert not (target / 'new').exists()
    assert not list(tmp_path.glob('.viame-addon-*'))


def test_cancel_download_removes_partial_archive(tmp_path, monkeypatch):
    import io
    cancel = tmp_path / 'cancel'
    monkeypatch.setenv('VIAME_ADDON_CANCEL_FILE', str(cancel))
    class Response(io.BytesIO):
        headers = {'Content-Length': '2000000'}
        def read(self, size):
            cancel.touch()
            return super().read(size)
    monkeypatch.setattr(m.urllib.request, 'urlopen', lambda *args, **kwargs: Response(b'x' * 2000000))
    download_dir = tmp_path / 'download'
    download_dir.mkdir()
    monkeypatch.setattr(m.tempfile, 'mkdtemp', lambda **kwargs: str(download_dir))
    addon = m.Addon('fish', 'https://example.test/fish.zip', '', '', '', [], '')
    with pytest.raises(m.InstallationCancelled):
        m.install_addon(tmp_path, addon)
    assert not download_dir.exists()


def test_cancel_cli_returns_distinct_exit_code(tmp_path, monkeypatch):
    cancel = tmp_path / 'cancel'
    cancel.touch()
    monkeypatch.setenv('VIAME_ADDON_CANCEL_FILE', str(cancel))
    csv = tmp_path / 'addons.csv'
    csv.write_text('FISH,https://example.test/fish.zip,Fish,,ALL-PLATFORMS,,fish.pipe\n')
    assert m.main(['--install-dir', str(tmp_path), '--csv', str(csv), 'install', 'FISH']) == 130


@pytest.mark.parametrize('cancel_download', [False, True])
def test_google_drive_progress_and_cancellation(tmp_path, monkeypatch, capsys, cancel_download):
    import json
    import sys
    import types
    cancel = tmp_path / 'cancel'
    monkeypatch.setenv('VIAME_ADDON_CANCEL_FILE', str(cancel))
    monkeypatch.setenv('VIAME_ADDON_PROGRESS', '1')
    def fake_download(*, url, output, quiet, use_cookies, progress):
        assert url == 'https://drive.google.com/file/d/public-id/view'
        assert quiet and not use_cookies
        output.write(b'archive')
        if cancel_download:
            cancel.touch()
        progress(7, 14)
        output.write(b'archive')
        return output
    monkeypatch.setitem(sys.modules, 'gdown', types.SimpleNamespace(download=fake_download))
    if cancel_download:
        with pytest.raises(m.InstallationCancelled):
            m.download('https://www.drive.google.com/file/d/public-id/view', tmp_path / 'pack.zip')
    else:
        m.download('https://www.drive.google.com/file/d/public-id/view', tmp_path / 'pack.zip')
        events = [json.loads(line.split(' ', 1)[1]) for line in capsys.readouterr().out.splitlines()
                  if line.startswith('VIAME_ADDON_PROGRESS ')]
        assert events[-1] == {'phase': 'download', 'done': 14, 'total': 14}
        assert (tmp_path / 'pack.zip').read_bytes() == b'archivearchive'
