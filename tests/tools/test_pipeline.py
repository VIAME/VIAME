import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_pipeline", Path(__file__).resolve().parents[2] / "tools" / "pipeline.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_repeated_include_is_not_a_cycle(tmp_path):
    p = tmp_path / 'main.pipe'
    common = tmp_path / 'common.pipe'
    common.write_text('config global\n  value = 1\n')
    p.write_text('include common.pipe\ninclude common.pipe\n')
    assert not m.load(p).errors
    common.write_text('include main.pipe\n')
    assert any('cycle' in e for e in m.load(p).errors)


def test_repeated_setting_preserves_comment(tmp_path):
    p = tmp_path / 'main.pipe'
    p.write_text('config global\n  value = 1 # comment\n')
    assert m.main(['set', str(p), '-s', 'global:value=long', '-s', 'global:value=2']) == 0
    assert p.read_text() == 'config global\n  value = 2 # comment\n'


# `pipeline check` asks `viame pipe-check` whether the names in a pipeline
# exist. These fake its report, so they need no install.
RESOLVABLE_PIPE = """\
process detector
  :: image_object_detector
  :detector:type                               no_such_detector
"""


def fake_report(resolved=True, status="ok", message=""):
    return {
        "status": status,
        "message": message,
        "processes": {
            "detector": {
                "type": "image_object_detector",
                "algos": {"detector:type": {"impl": "no_such_detector",
                                            "resolved": resolved}},
            }
        } if status == "ok" else {},
        "algos": {},
    }


def test_an_unregistered_implementation_is_an_error_at_its_process(tmp_path):
    p = tmp_path / "main.pipe"
    p.write_text(RESOLVABLE_PIPE)
    errors = m.resolution_errors(m.load(p), p, fake_report(resolved=False))
    assert len(errors) == 1
    assert errors[0].startswith(f"{p.resolve()}:1: detector:type selects \"no_such_detector\"")


def test_a_pipeline_that_does_not_build_is_an_error_without_the_source_location(tmp_path):
    p = tmp_path / "main.pipe"
    p.write_text(RESOLVABLE_PIPE)
    report = fake_report(status="error", message=(
        "There is no such process of type 'collate' in the registry, "
        "thrown from /src/process_factory.cxx:42"))
    errors = m.resolution_errors(m.load(p), p, report)
    assert errors == [f"{p}: There is no such process of type 'collate' in the registry"]


def test_check_fails_on_an_unresolved_name_and_passes_on_a_resolved_one(tmp_path, capsys):
    p = tmp_path / "main.pipe"
    p.write_text(RESOLVABLE_PIPE)
    with patch.object(m, "find_viame_executable", return_value="viame"), \
         patch.object(m, "run_pipe_check", return_value=fake_report(resolved=False)):
        assert m.main(["check", str(p)]) == 1
    assert 'selects "no_such_detector"' in capsys.readouterr().out
    with patch.object(m, "find_viame_executable", return_value="viame"), \
         patch.object(m, "run_pipe_check", return_value=fake_report(resolved=True)):
        assert m.main(["check", str(p)]) == 0


def test_no_resolve_does_not_ask_the_registry(tmp_path):
    p = tmp_path / "main.pipe"
    p.write_text(RESOLVABLE_PIPE)
    with patch.object(m, "run_pipe_check") as run:
        assert m.main(["check", "--no-resolve", str(p)]) == 0
    run.assert_not_called()


def test_without_viame_the_names_are_a_warning_not_a_failure(tmp_path, capsys):
    p = tmp_path / "main.pipe"
    p.write_text(RESOLVABLE_PIPE)
    with patch.object(m, "find_viame_executable", return_value=None):
        assert m.main(["check", str(p)]) == 0
    assert "not checked: no viame executable" in capsys.readouterr().out
