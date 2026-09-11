"""CPU regression checks for the RF-DETR/netharn training boundary.

Load the production methods without package-level imports so these checks do
not require KWIVER, torchvision, a GPU, or downloaded RF-DETR weights.
"""
import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock
import os
import sys
import warnings

import pytest

torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[3] / 'plugins/pytorch'


def load_method(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def test_weighted_loss_and_gradient_are_counted_once():
    forward = load_method('netharn/detection_models/rf_detr_models.py', 'forward', {
        'torch': torch, 'OrderedDict': OrderedDict,
        '_batch_to_rfdetr_inputs': lambda batch: batch,
    })
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    model = Mock()
    model.parameters = lambda: iter([parameter])
    model.return_value = parameter
    criterion = Mock(return_value={
        'loss_ce': parameter.square(), 'loss_bbox': parameter * 3,
        'class_error': parameter * 100,
    })
    criterion.weight_dict = {'loss_ce': 2, 'loss_bbox': 5}
    wrapper = SimpleNamespace(model=model, criterion=criterion,
                              input_norm=torch.nn.Identity())
    output = forward(wrapper, {
        'images': torch.zeros(1, 3, 8, 8), 'targets': [{}], 'image_size': (8, 8),
    }, return_result=False)
    loss = sum(output['loss_parts'].values())
    assert loss.item() == pytest.approx(38)
    loss.backward()
    assert parameter.grad.item() == pytest.approx(23)


@pytest.mark.parametrize('reduction, expected', [('mean', 3), ('sum', 6)])
def test_replica_loss_reduction(monkeypatch, reduction, expected):
    containers = ModuleType('viame.pytorch.netharn.data.data_containers')
    containers.BatchContainer = type('BatchContainer', (), {})
    monkeypatch.setitem(sys.modules, containers.__name__, containers)
    run_batch = load_method('netharn/detect_fit.py', 'run_batch', {'warnings': warnings})
    values = torch.tensor([2., 4.], requires_grad=True)
    harn = SimpleNamespace(
        _draw_timer=SimpleNamespace(toc=lambda: 0), batch_index=0,
        script_config={'num_draw': 0, 'draw_interval': 0},
        raw_model=SimpleNamespace(__BUILTIN_CRITERION__=True, __LOSS_REDUCTION__=reduction),
        model=SimpleNamespace(forward=lambda *a, **kw: {'loss_parts': {'loss': values}}),
    )
    _, parts = run_batch(harn, {})
    assert parts['loss'].item() == expected
    parts['loss'].backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 0.5 if reduction == 'mean' else 1))


@pytest.mark.parametrize('steps', [1, 4])
def test_accumulation_matches_single_batch(steps):
    backprop = load_method('netharn/fit_harn.py', 'backpropogate', {
        'torch': torch, 'profiler': SimpleNamespace(IS_PROFILING=False),
    })
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    harn = SimpleNamespace(
        raw_model=SimpleNamespace(__LOSS_REDUCTION__='mean'),
        dynamics={'batch_step': steps, 'grad_norm_max': None},
        preferences={'log_gradients': False}, current_tag='train', iter_index=0,
        optimizer=torch.optim.SGD([parameter], lr=0.1),
    )
    for index in range(steps):
        backprop(harn, index, None, parameter.square())
    assert parameter.item() == pytest.approx(1.6)


@pytest.mark.parametrize('arch, normalization, clipping', [
    ('rfdetr_large', 'imagenet', '0.1'),
    ('rf_detr_large', 'imagenet', '0.1'),
    ('yolo2', 'True', None),
])
def test_launcher_training_settings(arch, normalization, clipping):
    popen = Mock(return_value=SimpleNamespace(wait=lambda: None))
    namespace = {
        'TrainDetector': object, 'os': os,
        'subprocess': SimpleNamespace(Popen=popen),
        'threading': SimpleNamespace(current_thread=lambda: SimpleNamespace()),
    }
    init = load_method('netharn_trainer.py', '__init__', namespace)
    is_detr = load_method('netharn_trainer.py', '_is_detr_arch', namespace)
    update = load_method('netharn_trainer.py', 'update_model', namespace)
    trainer = SimpleNamespace()
    init(trainer)
    trainer._arch = arch
    trainer._gpu_count = 1
    trainer._no_format = True
    trainer._training_file = 'train.json'
    trainer._validation_file = 'val.json'
    trainer._is_detr_arch = lambda: is_detr(trainer)
    trainer.get_output_map = lambda: {}
    update(trainer)
    command = popen.call_args.args[0]
    assert '--normalize_inputs=' + normalization in command
    clips = [arg for arg in command if arg.startswith('--grad_norm_max=')]
    assert clips == ([] if clipping is None else ['--grad_norm_max=' + clipping])
