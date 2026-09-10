import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip('map_boxes')
pytest.importorskip('ensemble_boxes')

spec = importlib.util.spec_from_file_location("review_ensemble", Path(__file__).resolve().parents[2] / "tools" / "ensemble.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import json
import sys


def test_sequence_split_is_disjoint_and_reproducible():
    training, validation = m.split_sequences(5, .4, 7)
    assert len(validation) == 2
    assert set(training).isdisjoint(validation)
    assert sorted(training + validation) == list(range(5))
    assert (training, validation) == m.split_sequences(5, .4, 7)
    assert m.split_sequences(1, .2, 7) == ([0], [])
    with pytest.raises(ValueError):
        m.split_sequences(5, float('nan'), 7)


def test_calibration_uses_only_training_sequences(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / ('seq%d.csv' % i)
        p.write_text('1,image.png,%d,0,0,20,20,0.9,-1,fish,0.9\n' % i)
        paths.append(str(p))
    output = tmp_path / 'fusion.pipe'
    report = tmp_path / 'report.json'
    argv = ['ensemble', '-truth', *paths, '-computed', *paths,
            '-computed', *paths, '-methods', 'nms', '-trials', '0',
            '-refine-iters', '0', '--calibrate', '-output-calibration',
            str(tmp_path / 'calibration.json'), '-output-config', str(output),
            '-output-json', str(report)]
    original = m.collect_calibration_tables
    observed = []
    def collect(truth, computed, *args):
        observed.extend(next(iter(seq)) for seq in truth)
        return original(truth, computed, *args)
    with patch.object(sys, 'argv', argv), patch.object(m, 'collect_calibration_tables', side_effect=collect):
        m.main()
    data = json.loads(report.read_text())
    train, val = m.split_sequences(3, .2, 0)
    assert observed == train
    assert data['split']['validation_sequences'] == [paths[i] for i in val]
    assert data['validation_map'] == pytest.approx(1)
    assert json.loads(Path(str(output) + '.split.json').read_text()) == data['split']
