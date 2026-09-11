import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location(
    "review_index",
    Path(__file__).resolve().parents[2] / "tools" / "index.py")
m = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
with patch.dict(sys.modules, {
        'viame': types.ModuleType('viame'),
        'viame.core': types.SimpleNamespace(
            index_descriptors=types.SimpleNamespace(
                list_index_bundles=lambda path: []))}):
    spec.loader.exec_module(m)


# The two tests that stood here checked the PostgreSQL backend: that an
# existing postgres index refused a switch to files, and that a read-only
# command did not initialise a server. Both went with the backend. What is
# worth keeping is that the commands still refuse rather than guess.

def test_hash_refuses_a_config_file(tmp_path):
    """A config only ever named a PostgreSQL descriptor source."""
    config = tmp_path / "index.json"
    config.write_text("{}")

    args = m.argparse.Namespace(
        quiet=True, hash_only=False, model_dir=None, bit_length=256,
        itq_iterations=100, random_seed=0, normalize=None,
        max_train_descriptors=100000, uuids_list=None,
        config=str(config), descriptor_file=None, kwiver_csv=False,
        output_dir=str(tmp_path))

    m.index_descriptors.load_config = lambda path: {}

    with pytest.raises(SystemExit, match='backend is gone'):
        m.cmd_hash(args)


def test_hash_needs_a_descriptor_file(tmp_path):
    args = m.argparse.Namespace(
        quiet=True, hash_only=False, model_dir=None, bit_length=256,
        itq_iterations=100, random_seed=0, normalize=None,
        max_train_descriptors=100000, uuids_list=None,
        config=None, descriptor_file=None, kwiver_csv=False,
        output_dir=str(tmp_path))

    with pytest.raises(SystemExit, match='descriptor file'):
        m.cmd_hash(args)
