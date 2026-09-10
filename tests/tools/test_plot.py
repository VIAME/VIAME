import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("review_plot", Path(__file__).resolve().parents[2] / "tools" / "plot.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

import json


def test_score_json_generates_expected_plot_types(tmp_path):
    p = tmp_path / 'metrics.json'
    p.write_text(json.dumps({'precision': 1, 'pr_curve': {'points': []}, 'roc_curve': {'points': []}}))
    names = ['plot_pr_curve', 'plot_roc_curve', 'plot_detection_metrics_summary', 'plot_mot_metrics_summary', 'plot_hota_metrics_summary', 'plot_id_switch_breakdown', 'plot_track_quality_summary']
    from contextlib import ExitStack
    with ExitStack() as stack:
        mocks = [stack.enter_context(patch.object(m, name)) for name in names]
        m.generate_all_plots_from_json(str(p), str(tmp_path))
        for mock in mocks:
            mock.assert_called_once()


def test_false_alarms_axis_is_not_clipped(tmp_path):
    fig, ax = m.plt.subplots()
    with patch.object(m.plt, 'subplots', return_value=(fig, ax)), patch.object(fig, 'savefig'):
        m.plot_roc_curve({'points': [{'false_alarms_per_frame': 4, 'true_positive_rate': .8}]}, str(tmp_path / 'roc.png'))
    assert ax.get_xlim()[1] > 4
    assert ax.get_xlabel() == 'False alarms per frame'
