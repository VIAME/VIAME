import scriptconfig as scfg


class TorchLiberatorCLIConfig(scfg.Config):
    __default__ = {
        'model': scfg.Value(None, help='path to a liberated model'),
        'weights': scfg.Value(None, help='path to a weight checkpoint file'),
        'info': scfg.Value(None, help='path to metdadata json file'),
        'dst': scfg.Value(None, help='output file name'),
    }


def torch_liberator_cli(cmdline=True, **kw):
    """

    kw = {
        'model': 'liberated-model-file.py',
        'weights': 'checkpoints/_epoch_00000010.pt',
        'info': 'train_info.json',
        'dst': '.',
    }
    """
    from os.path import exists, isdir, split, join
    from .deployer import _make_package_name2
    from .deployer import _package_deploy2

    config = TorchLiberatorCLIConfig(cmdline=True, default=kw)

    assert exists(config['model'])

    info = {
        'snap_fpath': config['weights'],
        'model_fpath': config['model'],
        'train_info_fpath': config['info'],
    }

    if config['dst'] is None:
        config['dst'] = '.'
    if isdir(config['dst']):
        deploy_name = _make_package_name2(info)
        deploy_fpath = join(config['dst'], deploy_name + '.zip')
    else:
        deploy_fpath = config['dst']

    assert deploy_fpath.endswith('.zip')
    dpath, deploy_fname = split(deploy_fpath)
    _package_deploy2(dpath, info, name=deploy_fname)
