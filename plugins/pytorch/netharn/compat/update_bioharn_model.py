from os.path import exists
from os.path import join
import scriptconfig as scfg
import torch
import ubelt as ub
import copy
import json


class UpdateBioharnConfig(scfg.Config):
    default = {
        'deployed': scfg.Path(None, help='path to torch_liberator zipfile to convert'),
        'use_cache': scfg.Path(False, help='do nothing if we already converted'),
        'out_dpath': scfg.Path(None, help='place to write the new model to. (uses a cache directory if unspecified)'),
    }


def _debug():
    from .util.util_girder import grabdata_girder
    import ubelt as ub
    models = grabdata_girder('https://viame.kitware.com/api/v1', '60b44157463dff1a6cb124f9', 'VIAME-Default-Models-v1.0.zip')
    zfile = ub.zopen(models)

    for name in zfile.namelist():
        if 'fish_no_motion_detector.zip' in name:
            extracted = zfile.zfile.extract(name)
            break

    extracted = '/home/joncrall/configs/pipelines/models/fish_no_motion_detector.zip'

    # from torch_liberator import DeployedModel
    # deployed = DeployedModel.coerce(extracted)
    # mod = ub.import_module_from_path('/home/joncrall/configs/pipelines/models/deploy_MM_CascadeRCNN_ckbbymjh_013_MGATEY/MM_CascadeRCNN_1ff002.py')
    config = {
        'deployed': extracted,
        'use_cache': False,
        'out_dpath': '/home/joncrall/configs/pipelines/models',
    }


def update_deployed_bioharn_model(config):
    """
    Simply put old weights in a new model.
    If the code hasn't functionally changed then this should work.

    CLI:
        python -m bioharn.compat.update_bioharn_model \
            --deployed=$HOME/.cache/bioharn/deploy_MM_CascadeRCNN_igyhuonn_060_QWZMNS_sealion_coarse.zip

        python -m bioharn.compat.update_bioharn_model \
            --deployed=/home/joncrall/Downloads/tmp/configs/pipelines/models/sea_lion_multi_class.zip \
            --out_dpath=/home/joncrall/Downloads/tmp/configs/pipelines/models/

        python -m bioharn.compat.update_bioharn_model \
            --deployed=/home/joncrall/Downloads/tmp/configs/pipelines/models/sea_lion_single_class.zip \
            --out_dpath=/home/joncrall/Downloads/tmp/configs/pipelines/models/

        python -m bioharn.detect_predict \
            --deployed=/home/joncrall/Downloads/tmp/configs/pipelines/models/sea_lion_multi_class_bio3x.zip \
            --out_dpath=/home/joncrall/Downloads/tmp/configs/pipelines/models/multi_3x \
            --dataset=$HOME/.cache/bioharn/sealion_test_img_2010.jpg \
            --draw=1

        python -m bioharn.detect_predict \
            --deployed=/home/joncrall/Downloads/tmp/configs/pipelines/models/sea_lion_multi_class.zip \
            --out_dpath=/home/joncrall/Downloads/tmp/configs/pipelines/models/multi_1x \
            --dataset=$HOME/.cache/bioharn/sealion_test_img_2010.jpg \
            --draw=1

        python -m bioharn.detect_predict \
            --deployed=/home/joncrall/Downloads/tmp/configs/pipelines/models/sea_lion_single_class.zip \
            --out_dpath=/home/joncrall/Downloads/tmp/configs/pipelines/models/single_1x \
            --dataset=$HOME/.cache/bioharn/sealion_test_img_2010.jpg \
            --draw=1

    """
    from torch_liberator import deployer
    from viame.pytorch import netharn as nh
    from ..detect_predict import setup_module_aliases
    from ..detection_models import mm_models

    # Set up module aliases for backwards compatibility with old models
    setup_module_aliases()

    config = UpdateBioharnConfig(config)

    deploy_fpath = config['deployed']

    if config['out_dpath'] is None:
        extract_dpath = ub.ensure_app_cache_dir('torch_liberator/extracted')
    else:
        extract_dpath = config['out_dpath']

    new_name = ub.augpath(deploy_fpath, dpath='', suffix='_bio4x')
    new_fpath = join(extract_dpath, new_name)

    print('Upgrade deployed model config: config = {!r}'.format(config))

    if config['use_cache']:
        if exists(new_fpath):
            print('Returning cached new_fpath = {!r}'.format(new_fpath))
            return new_fpath

    deployed = deployer.DeployedModel(deploy_fpath)

    print('Extracting old snapshot to: {}'.format(extract_dpath))
    temp_fpath = deployed.extract_snapshot(extract_dpath)

    model_cls, model_initkw = deployed.model_definition()

    new_initkw = model_initkw
    new_model = mm_models.MM_CascadeRCNN(**new_initkw)
    new_model._initkw = new_initkw

    xpu = nh.XPU.coerce('cpu')
    new_model_state = xpu.load(temp_fpath)

    # print(new_model.detector.roi_head.bbox_head[0].fc_cls.weight.shape)
    # print(model_state_2['bbox_head.0.fc_cls.weight'].shape)
    # print(new_model_state['state_dict']['roi_head.bbox_head.0.fc_cls.weight'].shape)

    # from viame.pytorch.netharn.initializers.functional import load_partial_state
    from torch_liberator.initializer import load_partial_state
    load_info = load_partial_state(new_model, new_model_state['model_state_dict'], verbose=3)
    del load_info
    # new_model_state['model_state_dict']['input_norm.mean']
    # _ = new_model.load_state_dict(new_model_state['model_state_dict'])

    TEST_FORWARD = 0
    if TEST_FORWARD:
        batch = {
            'inputs': {
                'rgb': torch.rand(1, 3, 256, 256),
            }
        }
        outputs = new_model.forward(batch, return_loss=False)
        batch_dets = new_model.coder.decode_batch(outputs)
        dets = batch_dets[0]
        print('dets = {!r}'.format(dets))

    new_train_info = copy.deepcopy(deployed.train_info())
    new_train_info['__mmdet_conversion__'] = '1x_to_2x'
    new_train_info['__bioharn_model_vesion__'] = new_model.__bioharn_model_vesion__

    new_train_info_fpath = join(extract_dpath, 'train_info.json')
    new_snap_fpath = temp_fpath
    with open(new_train_info_fpath, 'w') as file:
        json.dump(new_train_info, file, indent='    ')

    new_snapshot = {
        'model_state_dict': new_model.state_dict(),
        'epoch': new_model_state['epoch'],
        '__mmdet_conversion__': '1x_to_2x',
    }
    torch.save(new_snapshot, new_snap_fpath)

    new_deployed = deployer.DeployedModel.custom(
        model=new_model, snap_fpath=new_snap_fpath,
        train_info_fpath=new_train_info_fpath, initkw=new_initkw)

    new_name = ub.augpath(deploy_fpath, dpath='', suffix='_bio3x')
    fpath = new_deployed.package(dpath=extract_dpath, name=new_name)
    print('fpath = {!r}'.format(fpath))

    ub.delete(new_snap_fpath)
    ub.delete(new_train_info_fpath)

    return fpath


def main():
    config = UpdateBioharnConfig(cmdline=True)
    update_deployed_bioharn_model(config)


if __name__ == '__main__':
    """
    CommandLine:
        python -m bioharn.compat.update_bioharn_model
    """
    main()
