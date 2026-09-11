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


class LabelContainer:
    def __init__(self, data):
        self.data = data


@pytest.mark.parametrize('collated', [False, True])
def test_masks_and_keypoints_follow_filtered_boxes(collated):
    convert = load_method('netharn/detection_models/rf_detr_models.py',
                          '_batch_to_rfdetr_targets', {
        'torch': torch, 'data_containers': SimpleNamespace(BatchContainer=LabelContainer),
    })
    fields = {
        'tlbr': [torch.tensor([[0, 0, 8, 4], [0, 0, 4, 2]]), torch.empty(0, 4)],
        'class_idxs': [torch.tensor([1, 0]), torch.empty(0, dtype=torch.long)],
        'weight': [torch.tensor([1., 0.]), torch.empty(0)],
        'class_masks': [torch.ones(2, 4, 8), torch.empty(0, 4, 8)],
        'has_mask': [torch.tensor([1, -1]), torch.empty(0)],
        'keypoints': [torch.tensor([[[4, 2, 2], [0, 0, 0]],
                                    [[1, 1, 2], [2, 1, 2]]]), torch.empty(0, 2, 3)],
    }
    if collated:
        fields = {k: LabelContainer([[v[0]], [v[1]]]) for k, v in fields.items()}
    targets = convert({'label': fields}, (4, 8))
    assert targets[0]['labels'].tolist() == [1]
    torch.testing.assert_close(targets[0]['boxes'], torch.tensor([[.5, .5, 1, 1]]))
    torch.testing.assert_close(targets[0]['keypoints'], torch.tensor([[[.5, .5, 2], [0, 0, 0]]]))
    assert targets[0]['masks'].shape == (1, 4, 8)
    assert targets[0]['masks'].dtype == torch.bool
    assert targets[1]['masks'].shape == (0, 4, 8)
    assert targets[1]['keypoints'].shape == (0, 2, 3)


def test_unannotated_masks_are_not_trained_as_foreground():
    convert = load_method('netharn/detection_models/rf_detr_models.py',
                          '_batch_to_rfdetr_targets', {
        'torch': torch, 'data_containers': SimpleNamespace(BatchContainer=LabelContainer),
    })
    with pytest.raises(ValueError, match='requires a mask'):
        convert({'label': {'cxywh': [torch.ones(1, 4)], 'class_idxs': [torch.tensor([0])],
                          'class_masks': [torch.full((1, 4, 8), 2)],
                          'has_mask': [torch.tensor([-1])]}}, (4, 8))


def test_keypoint_slots_after_geometric_transform():
    import numpy as np
    kwimage = pytest.importorskip('kwimage')
    pack = load_method('netharn/detect_dataset.py', '_keypoint_targets', {'torch': torch, 'np': np})
    pts = kwimage.Points.from_coco([
        {'xy': [2, 3], 'keypoint_category': 'Tail', 'visible': 2},
        {'xy': [1, 2], 'keypoint_category': 'HEAD', 'visible': 1},
    ], classes=['Tail', 'HEAD'])
    dets = kwimage.Detections(boxes=kwimage.Boxes([[0, 0, 5, 5], [0, 0, 5, 5]], 'xywh'),
                              keypoints=kwimage.PointsList([pts, None]))
    dets = dets.scale(2).translate((1, 1))
    values = pack(dets, ['head', 'tail'], (8, 8))
    torch.testing.assert_close(values[0], torch.tensor([[3., 5., 1.], [5., 7., 2.]]))
    assert not values[1].any()
    # Cropped-out points have absent visibility, rather than clamped coordinates.
    assert not pack(dets, ['head', 'tail'], (6, 8))[0, 1].any()


@pytest.fixture
def native_config(monkeypatch):
    """Load real vendored config classes without RF-DETR's torchvision imports."""
    import importlib.util
    root = ROOT.parents[1] / 'packages/pytorch-libs/rf-detr/src/rfdetr'
    package = ModuleType('rfdetr')
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, 'rfdetr', package)
    utilities = ModuleType('rfdetr.utilities')
    utilities.__path__ = [str(root / 'utilities')]
    monkeypatch.setitem(sys.modules, 'rfdetr.utilities', utilities)
    spec = importlib.util.spec_from_file_location('rfdetr.config', root / 'config.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'rfdetr.config', module)
    spec.loader.exec_module(module)
    package.config = module
    return module


@pytest.mark.parametrize('segmentation, expected_queries, patch, layers', [
    (False, 300, 16, 4), (True, 200, 12, 5),
])
def test_actual_variant_configuration(native_config, segmentation, expected_queries, patch, layers):
    configure = load_method('netharn/detection_models/rf_detr_models.py', '_get_variant_config', {'torch': torch})
    wrapper = SimpleNamespace(segmentation_head=segmentation, num_classes=2, keypoint_names=['head', 'tail'])
    config = configure(wrapper, 'large', None, (960, 1728))
    assert config.num_queries == expected_queries
    assert config.patch_size == patch
    assert config.dec_layers == layers
    assert config.keypoint_head and config.num_keypoints == 2
    assert config.resolution == (960, 1728)
    assert config.positional_encoding_size == (960 // patch, 1728 // patch)
    with pytest.raises(ValueError):
        configure(wrapper, 'large', None, (961, 1728))


def test_native_weights_use_rfdetr_loader(native_config, monkeypatch, tmp_path):
    path = tmp_path / 'native.pth'
    path.touch()
    models = ModuleType('rfdetr.models')
    network = torch.nn.Linear(2, 2)
    models.build_model_from_config = Mock(return_value=network)
    models.load_pretrain_weights = Mock(return_value=['fish'])
    models.build_criterion_from_config = Mock(return_value=('criterion', 'postprocess'))
    assets = ModuleType('rfdetr.assets.model_weights')
    assets.get_model_cache_dir = Mock(return_value=str(tmp_path))
    monkeypatch.setitem(sys.modules, models.__name__, models)
    monkeypatch.setitem(sys.modules, assets.__name__, assets)
    build = load_method('netharn/detection_models/rf_detr_models.py', '_build_model', {})
    config = native_config.RFDETRSegLargeConfig(num_classes=1)
    result = build(SimpleNamespace(segmentation_head=True, classes=['fish']), config, str(path))
    assert result == (network, 'criterion', 'postprocess')
    assert config.pretrain_weights == str(path)
    models.load_pretrain_weights.assert_called_once_with(network, config)
    assert isinstance(models.build_model_from_config.call_args.args[1], native_config.SegmentationTrainConfig)


def test_launcher_native_seed_and_keypoints(tmp_path):
    popen = Mock(return_value=SimpleNamespace(wait=lambda: None))
    namespace = {'TrainDetector': object, 'os': os, 'subprocess': SimpleNamespace(Popen=popen),
                 'threading': SimpleNamespace(current_thread=lambda: SimpleNamespace())}
    init = load_method('netharn_trainer.py', '__init__', namespace)
    update = load_method('netharn_trainer.py', 'update_model', namespace)
    trainer = SimpleNamespace()
    init(trainer)
    path = tmp_path / 'seg.pth'
    path.touch()
    trainer._native_seed_model = str(path)
    trainer._keypoints = trainer._segmentation_head = True
    trainer._is_detr_arch = lambda: True
    trainer._gpu_count = 1
    trainer._no_format = True
    trainer._training_file, trainer._validation_file = 'train.json', 'val.json'
    trainer.get_output_map = lambda: {}
    update(trainer)
    cmd = popen.call_args.args[0]
    assert '--native_seed_model=' + str(path) in cmd
    assert '--keypoints=True' in cmd
    assert '--keypoint_names=head,tail' in cmd
    assert '--segmentation_head=True' in cmd
    assert not any(arg.startswith('--pretrained=') for arg in cmd)
    trainer._seed_model = 'netharn.zip'
    with pytest.raises(ValueError, match='either seed_model'):
        update(trainer)
    trainer._seed_model = ''
    path.unlink()
    with pytest.raises(FileNotFoundError):
        update(trainer)


def test_decoder_preserves_masks_and_keypoints_with_score_filter():
    import numpy as np
    kwimage = pytest.importorskip('kwimage')
    decode = load_method('netharn/detection_models/rf_detr_models.py', 'decode_batch', {
        'torch': torch, 'np': np, 'kwimage': kwimage,
        'data_containers': SimpleNamespace(BatchContainer=LabelContainer),
    })
    coder = SimpleNamespace(classes=['fish'], keypoint_names=['head', 'tail'], score_thresh=.5)
    result = {'scores': torch.tensor([.2, .9]), 'labels': torch.tensor([0, 0]),
              'boxes': torch.tensor([[0, 0, 1, 1], [0, 0, 4, 4]]),
              'masks': torch.ones(2, 1, 4, 4),
              'keypoints': torch.tensor([[[0, 0, .2], [0, 0, .3]],
                                         [[1, 2, .8], [3, 2, .9]]])}
    det, = decode(coder, {'batch_results': [result]})
    assert len(det) == 1
    assert det.data['segmentations'][0].to_mask().data.all()
    np.testing.assert_allclose(det.data['keypoints'][0].xy, [[1, 2], [3, 2]])


def test_keypoint_run_rejects_missing_or_wrong_names():
    import numpy as np
    kwimage = pytest.importorskip('kwimage')
    kwcoco = pytest.importorskip('kwcoco')
    namespace = {'torch': torch, 'np': np, 'kwimage': kwimage}
    load_method('netharn/detect_dataset.py', '_keypoint_targets', namespace)
    validate = load_method('netharn/detect_dataset.py', '_validate_keypoint_training_data', namespace)
    dset = kwcoco.CocoDataset()
    cid = dset.add_category('fish')
    gid = dset.add_image(file_name='unused.png', width=8, height=8)
    aid = dset.add_annotation(image_id=gid, category_id=cid, bbox=[0, 0, 8, 8])
    with pytest.raises(ValueError, match='no visible'):
        validate(dset, ['head', 'tail'])
    dset.anns[aid]['keypoints'] = [{'xy': [2, 3], 'keypoint_category': 'HEAD', 'visible': 2}]
    validate(dset, ['head', 'tail'])
    with pytest.raises(ValueError, match='no visible'):
        validate(dset, ['nose'])


@pytest.mark.parametrize('lightning', [False, True])
def test_real_native_checkpoint_keeps_head_and_query_weights(native_config, monkeypatch, tmp_path, lightning):
    import importlib.util
    assets = ModuleType('rfdetr.assets.model_weights')
    assets.download_pretrain_weights = lambda *a, **kw: None
    assets.validate_pretrain_weights = lambda *a, **kw: True
    assets.get_model_cache_dir = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, assets.__name__, assets)
    root = ROOT.parents[1] / 'packages/pytorch-libs/rf-detr/src/rfdetr'
    decorators = ModuleType('rfdetr.utilities.decorators')
    decorators.deprecated = lambda *a, **kw: lambda func: func
    monkeypatch.setitem(sys.modules, decorators.__name__, decorators)
    spec = importlib.util.spec_from_file_location('native_weights_test', root / 'models/weights.py')
    weights = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(weights)
    model = torch.nn.Module()
    model.class_embed = torch.nn.Linear(4, 2)
    model.query_feat = torch.nn.Embedding(200, 4)
    model.refpoint_embed = torch.nn.Embedding(200, 4)
    # A new keypoint head must keep its initialized weights when seeding from seg.
    model.keypoint_head = torch.nn.Linear(4, 6)
    initial_keypoints = model.keypoint_head.weight.detach().clone()
    state = {'class_embed.weight': torch.full((2, 4), 3.),
             'class_embed.bias': torch.full((2,), 2.),
             'query_feat.weight': torch.arange(2600 * 4).reshape(2600, 4).float(),
             'refpoint_embed.weight': torch.ones(2600, 4)}
    args = {'num_queries': 200, 'group_detr': 13, 'class_names': ['fish']}
    checkpoint = ({'state_dict': {'model.' + k: v for k, v in state.items()},
                   'hyper_parameters': args} if lightning else {'model': state, 'args': args})
    path = tmp_path / ('seed.ckpt' if lightning else 'seed.pth')
    torch.save(checkpoint, path)
    config = native_config.RFDETRSegLargeConfig(
        num_classes=1, group_detr=1, pretrain_weights=str(path), keypoint_head=True)
    names = weights.load_pretrain_weights(model, config)
    assert names == ['fish']
    torch.testing.assert_close(model.class_embed.weight, state['class_embed.weight'])
    torch.testing.assert_close(model.query_feat.weight, state['query_feat.weight'][:200])
    torch.testing.assert_close(model.keypoint_head.weight, initial_keypoints)
