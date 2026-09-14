"""
Deployment component of the Pytorch exporter.

This file contains DeployedModel, which consists of logic to take the
model topology definition along with the "best" snapshot in a training
directory and package it up into a standalone zipfile. The DeployedModel can
also be used to reload model from this zipfile. Thus this zipfile can be passed
around as a pytorch model topology+pretrained weights transfer format.

The following docstring illustrates how this module may be used.

CommandLine:
    # Runs the following example
    xdoctest -m torch_liberator.deployer __doc__:0

    # Runs all the doctests
    xdoctest -m torch_liberator.deployer all


# TODO: be a better example for the standalone package


Example:
    >>> # xdoctest: +REQUIRES(module:torchvision)
    >>> from torch_liberator.deployer import *
    >>> import torch_liberator
    >>> from torchvision import models

    >>> print('--- DEFINE A MODEL ---')
    >>> model = models.alexnet(pretrained=False)  # false for test speed
    >>> initkw = dict(num_classes=1000)  # not all models nicely supply this
    >>> model._initkw = initkw

    >>> print('--- DEPLOY THE MODEL ---')
    >>> zip_fpath = torch_liberator.deploy(model, 'test-deploy.zip')

    >>> print('--- LOAD THE DEPLOYED MODEL ---')
    >>> from torch_liberator.deployer import DeployedModel
    >>> loader = DeployedModel(zip_fpath)
    >>> model = loader.load_model()
"""
import glob
import json
import ubelt as ub
import zipfile
import os
from os.path import basename, exists, isdir, join, relpath

__all__ = ['DeployedModel', 'deploy']


def deploy(model, fpath, initkw=None, export_modules=[]):
    """
    Create a deployfile from a Pytorch model

    Args:
        model (torch.Module): this is a torch model, which ideally is
            statically parsable. The value of modgt

        fpath (str): file path to the deployfile to be written

        initkw: overwrite model._initkw if it exists, otherwise you must
            specify it here. These are the args used to create the instance of
            your model.

    Returns:
        str: fpath the path to the file

    Notes:
        We accept most models, but there are known examples where things break:
            (e.g. mmdet).

    """
    import shutil
    import tempfile
    import torch
    from . import exporter

    temp_dpath = tempfile.mkdtemp()
    snap_fpath = join(temp_dpath, 'deploy_snapshot.pt')
    torch.save(model.state_dict(), snap_fpath)

    if initkw is None:
        initkw = getattr(model, '_initkw', initkw)

    model_fpath = exporter.export_model_code(
        temp_dpath, model.__class__, initkw=initkw,
        export_modules=export_modules
    )
    deployed = DeployedModel.custom(snap_fpath, model_fpath)

    name = basename(fpath)
    temp_deploy_fpath = deployed.package(temp_dpath, name=name)

    shutil.move(temp_deploy_fpath, fpath)
    return fpath


def existing_snapshots(train_dpath):
    # NOTE: Specific to netharn directory structure
    import parse
    snapshot_dpath = join(train_dpath, 'torch_snapshots/')
    snapshots = {}
    if exists(snapshot_dpath):
        prev_states = sorted(glob.glob(join(snapshot_dpath, '_epoch_*.pt')))
        snapshots.update({
            parse.parse('{}_epoch_{num:d}.pt', path).named['num']: path
            for path in prev_states})

    snapshot_dpath = join(train_dpath, 'checkpoints/')
    if exists(snapshot_dpath):
        prev_states = sorted(glob.glob(join(snapshot_dpath, '_epoch_*.pt')))
        snapshots.update({
            parse.parse('{}_epoch_{num:d}.pt', path).named['num']: path
            for path in prev_states})

    snapshot_dpath = join(train_dpath, 'checkpoints/')
    return snapshots


def find_best_snapshot(train_dpath):
    """
    Returns snapshot written by monitor if available otherwise takes the last
    one.
    """
    # NOTE: Specific to netharn directory structure
    # Netharn should populate best_snapshot.pt if there is a validation set.
    # Other names are to support older codebases.
    expected_names = [
        'best_snapshot.pt',
        'best_snapshot2.pt',
        'final_snapshot.pt',
        'deploy_snapshot.pt',
    ]
    for snap_fname in expected_names:
        snap_fpath = join(train_dpath, snap_fname)
        if exists(snap_fpath):
            break

    if not exists(snap_fpath):
        snap_fpath = None

    if not snap_fpath:
        epoch_to_fpath = existing_snapshots(train_dpath)
        if epoch_to_fpath:
            snap_fpath = epoch_to_fpath[max(epoch_to_fpath)]
    return snap_fpath


def unpack_model_info(path):
    """
    return paths to the most relevant files in a zip or path deployment.

    If path is not a zipfile, this function expects a netharn fit directory
    structure.

    Args:
        path (PathLike): either a zip deployment or train_dpath.
            Preferably this is a zip deployment file or a path to an unzipped
            deploy file. If this is a train_dpath, then it should at least
            contain a model topology py file and snapshot pt file, otherwise
            subsequent usage will likely fail.
    """
    info = {
        'train_info_fpath': None,
        'snap_fpath': None,
        'model_fpath': None,

        # TODO: need to rename and allow a list of arbitrary files
        'glance': [],  # a list of files in the glance directory
    }
    def populate(root, fpaths):
        # TODO: make more robust
        for fpath in fpaths:
            # FIXME: make this more general and robust
            if fpath.endswith('.json'):
                info['train_info_fpath'] = join(root, fpath)
            if fpath.endswith('.pt'):
                info['snap_fpath'] = join(root, fpath)
            if fpath.endswith('.py'):
                new_fpath = join(root, fpath)
                if info['model_fpath'] is not None:
                    try:
                        # Try to take the most recent path if possible.
                        # This will fail if the file is in a zipfile
                        # (because we should not package multiple models)
                        cur_time = os.stat(info['model_fpath']).st_mtime
                        new_time = os.stat(new_fpath).st_mtime
                        if new_time < cur_time:
                            continue  # Keep the current path
                    except OSError:
                        raise Exception(
                            'Multiple model paths! {} and {}'.format(
                                info['model_fpath'], fpath))
                info['model_fpath'] = new_fpath
            # TODO: make including arbitrary files easier
            if fpath.startswith(('glance/', 'glance\\')):
                info['glance'].append(join(root, fpath))

    if path.endswith('.zip'):
        zipfpath = path
        myzip = zipfile.ZipFile(zipfpath, 'r')
        with zipfile.ZipFile(zipfpath, 'r') as myzip:
            populate(zipfpath, (f.filename for f in myzip.filelist))

    elif exists(path) and isdir(path):
        # Populate core files
        populate(path, os.listdir(path))
        # Populate extra glanceable files
        populate(path, [
            relpath(p, path) for p in glob.glob(join(path, 'glance/*'))])
        # If there are no snapshots in the root directory, then
        # use the latest snapshot from the torch_snapshots dir
        if info['snap_fpath'] is None:
            info['snap_fpath'] = find_best_snapshot(path)

    else:
        raise ValueError('cannot unpack model ' + path)
    return info


def _make_package_name2(info):
    """
    Construct a unique and descriptive name for the deployment

    This generally relies on netharn style info
    """
    snap_fpath = info['snap_fpath']
    model_fpath = info['model_fpath']
    train_info_fpath = info['train_info_fpath']

    if train_info_fpath and exists(train_info_fpath):
        train_info = json.load(open(train_info_fpath, 'r'))
        # model_name = train_info['hyper']['model'][0].split('.')[-1]
        if 'name' in train_info:
            run_name = train_info['name']
        elif 'nice' in train_info:
            run_name = train_info['nice']
        else:
            run_name = 'UNKNOWN-RUN'
        if 'train_hashid' in train_info:
            train_hash = train_info['train_hashid']
        else:
            train_hash = ub.hash_data(train_info['train_id'], hasher='sha512',
                                      base='abc', types=True)[0:8]
    else:
        run_name = os.path.splitext(os.path.basename(model_fpath))[0]
        train_hash = 'UNKNOWN-TRAINID'
        print('WARNING: Train info metadata does not exist')

    try:
        # netharn models contain epoch info in the weights file
        import torch
        state = torch.load(snap_fpath,
                           map_location=lambda storage, location: storage, weights_only=False)
        epoch = '{:03d}'.format(state['epoch'])
    except Exception:
        epoch = 'UNKNOWN-EPOCH'

    weights_hash = ub.hash_file(snap_fpath, base='abc',
                                hasher='sha512')[0:6].upper()

    deploy_name = 'deploy_{run}_{trainid}_{epoch}_{weights}'.format(
        run=run_name, trainid=train_hash, epoch=epoch,
        weights=weights_hash)
    return deploy_name


def _package_deploy2(dpath, info, name=None):
    """
    Combine the model, weights, and info files into a single deployable file

    Args:
        dpath (PathLike): where to dump the deployment
        info (Dict): containing model_fpath and snap_fpath and optionally
            train_info_fpath and glance, which is a list of extra files.
        name (str, default=None): the name of the zipfile to deploy to.
            If not specified, one will be constructed.

    Ignore:
        dpath = '/home/joncrall/.cache/netharn/tests/_package_custom'
        path = '/home/joncrall/work/opir/fit/name/_Sim3-kw6-99-finetune_ML3D_BEST_2018-9-20_LR1e-4_f2_vel0.0_hn0.25_bs64_nr5.0'
        info = unpack_model_info(path)
        zipfpath = _package_deploy2(dpath, info)


    """
    model_fpath = info['model_fpath']
    snap_fpath = info['snap_fpath']
    train_info_fpath = info.get('train_info_fpath', None)

    if not snap_fpath:
        raise FileNotFoundError('No weights are associated with the model')

    if name is None:
        deploy_name = _make_package_name2(info)
        deploy_fname = deploy_name + '.zip'
    else:
        if not name.endswith('.zip'):
            raise ValueError('The deployed package name must end in .zip')
        deploy_name = os.path.splitext(name)[0]
        deploy_fname = name

    def zwrite(myzip, fpath, fname=None):
        if fname is None:
            fname = relpath(fpath, dpath)
        myzip.write(fpath, arcname=join(deploy_name, fname))

    zipfpath = join(dpath, deploy_fname)
    with zipfile.ZipFile(zipfpath, 'w') as myzip:
        if train_info_fpath and exists(train_info_fpath):
            zwrite(myzip, train_info_fpath, fname='train_info.json')
        zwrite(myzip, snap_fpath, fname='deploy_snapshot.pt')
        zwrite(myzip, model_fpath, fname=os.path.basename(model_fpath))
        # Add some quick glanceable info
        for p in info.get('glance', []):
            zwrite(myzip, p, fname=join('glance', os.path.basename(p)))
        # for bestacc_fpath in glob.glob(join(train_dpath, 'best_epoch_*')):
        #     zwrite(myzip, bestacc_fpath)
        # for p in glob.glob(join(train_dpath, 'glance/*')):
        #     zwrite(myzip, p)
    print('[DEPLOYER] Deployed zipfpath={}'.format(zipfpath))
    return zipfpath


class DeployedModel(ub.NiceRepr):
    """
    Can setup an initializer and model from a deployed zipfile or a train path

    TODO:
        NEEDS TO BE REDESIGNED

        Optional Inputs:

            * Train info metadata files

        Possible Inputs:

            * An model instance

            * An model file and a weights file.

            * An deploy file

        Needs Netharn:
            * A training dpath (can infer model and weights file from this)

            * A weights file (can infer model file from this)

            * A model file (can infer weights file from this)

    CommandLine:
        xdoctest -m torch_liberator.deployer DeployedModel

    Example:
        >>> # xdoctest: +REQUIRES(module:netharn)
        >>> # Test the train folder as the model deployment
        >>> train_dpath = _demodata_trained_dpath()
        >>> self = DeployedModel(train_dpath)
        >>> model = self.load_model()
        ...
        >>> print('model.__module__ = {!r}'.format(model.__module__))

    Example:
        >>> # xdoctest: +REQUIRES(module:netharn)
        >>> # Test the zip file as the model deployment
        >>> zip_fpath = _demodata_zip_fpath()
        >>> self = DeployedModel(zip_fpath)
        >>> model = self.load_model()
        ...
        >>> # NOTE: the module name should be consistent, but due to
        >>> # small library changes it often changes, so we are permissive
        >>> # with this got/want test
        >>> print('model.__module__ = {!r}'.format(model.__module__))
        model.__module__ = 'deploy_demo_liberator_static_..._.../ToyNet2d_...'

        model.__module__ = 'deploy_ToyNet2d_mhuhweia_000_.../ToyNet2d_...'

        model.__module__ = 'deploy_ToyNet2d_rljhgepw_000_.../ToyNet2d_2a3f49'
    """
    def __init__(self, path):
        self.path = path
        self._model = None
        self._info = None

    def __nice__(self):
        return self.__json__()

    def __json__(self):
        if self.path is None:
            if self._info:
                return ub.urepr(self._info, nl=0)
        else:
            return self.path

    def package(self, dpath=None, name=None):
        """
        If self.path is a directory, packages important info into a deployable
        zipfile.

        Args:
            dpath (PathLike, optional): directory to dump your packaged model.
                If not specified, it uses the netharn train_dpath if available.
            name (str, default=None): the name of the zipfile to deploy to.
                If not specified, one will be constructed.

        Returns:
            PathLike: path to single-file deployment
        """
        if dpath is None:
            if self.path is None:
                raise ValueError('Must specify dpath for custom deployments')
            else:
                if self.path.endswith('.zip'):
                    raise Exception('Deployed model is already a package')
                dpath = self.path

        zip_fpath = _package_deploy2(dpath, self.info, name=name)
        return zip_fpath

    @property
    def info(self):
        if self._info is None:
            self._info = self.unpack_info()
        return self._info

    def unpack_info(self):
        return unpack_model_info(self.path)

    def model_definition(self):
        """
        Return the model class definition and the initkw use to construct it

        Returns:
            Tuple[type, dict]: (model_cls, initkw)
        """
        model_fpath = self.info['model_fpath']
        module = ub.import_module_from_path(model_fpath)

        export_version = getattr(module, '__pt_export_version__', '0')
        export_version = list(map(int, export_version.split('.')))
        if export_version >= [0, 2, 0]:
            model_cls = module.get_model_cls()
            initkw = module.get_initkw()
        else:
            # Hack to get information from older versions of pytorch_export
            import inspect
            # perhaps use xinspect
            from xdoctest import static_analysis
            print('Hacking to grab model_cls and initkw')
            model = module.make()
            model_cls = model.__class__
            source = inspect.getsource(module.make)
            print(source)
            initkw = static_analysis.parse_static_value('initkw', source=source)
            # Try to reconstruct initkw
        model_ = (model_cls, initkw)
        return model_

    def train_info(self):
        """
        Load any json metadata stored with the deployed model.

        This is the data referenced by `train_info_fpath`.
        """
        from .util import util_zip
        train_info_fpath = self.info.get('train_info_fpath', None)
        if train_info_fpath is not None:
            train_info = json.load(util_zip.zopen(train_info_fpath, 'r'))
        else:
            train_info = None
        return train_info

    def load_model(self):
        """
        Return an instance of the deployed model.

        If this structure contains a preconstructed instance, it is simply
        returned, otherwise a new instance of the model is created.
        """
        if self._model is not None:
            return self._model

        model_cls, model_kw = self.model_definition()
        model = model_cls(**model_kw)

        # Always load models onto the CPU first
        model = model.to('cpu')

        # TODO: need to update the load_partial_state function
        from .initializer import Pretrained
        initializer = Pretrained(fpath=self.info['snap_fpath'])
        initializer.forward(model)
        return model

    def extract_snapshot(self, extract_dpath=None):
        """
        If the weights snapshot is part of a zipfile, extract it to disk

        Returns:
            str : the path to the extracted weights
        """
        from .util.util_zip import split_archive
        import zipfile
        # Extract the snapshot fpath to disk
        snap_fpath = self.info['snap_fpath']
        archive_fpath, internal = split_archive(snap_fpath)
        if archive_fpath is None:
            raise Exception('deployed snapshot is not in an archive')
        if extract_dpath is None:
            extract_dpath = ub.ensure_app_cache_dir('torch_liberator/extracted')
        with zipfile.ZipFile(archive_fpath, 'r') as myzip:
            myzip.extract(internal, extract_dpath)
        temp_fpath = join(extract_dpath, internal)
        assert exists(temp_fpath)
        return temp_fpath

    @classmethod
    def ensure_mounted_model(cls, deployed, xpu=None, log=print):
        """
        Ensure that a deployed model is loaded and mounted.

        Helper method that can accept either a raw model or packaged deployed
        model is loaded and mounted on a specific XPU. This provides a one line
        solution for applications that may want to ensure that a model is
        mounted and ready for predict. When the model is already mounted this
        is very fast and just passes the data through. If the input is a
        packaged deployed file, then it does the required work to prep the
        model.

        Args:
            deployed (DeployedModel | PathLike | torch.nn.Module):
                either a packed deployed model, a path to a deployed model, or
                an already mounted torch Module.
            xpu (str | XPU): which device to mount on
            log (callable, optional): logging or print function

        Returns:
            Tuple[torch.nn.Module, XPU]:
                the mounted model, and the device it is mounted on.
        """
        # FIXME: netharn dependency
        import torch
        from . import xpu_device
        xpu = xpu_device.XPU.coerce(xpu)

        if isinstance(deployed, str):
            deployed = DeployedModel(deployed)

        if isinstance(deployed, torch.nn.Module):
            # User passed in the model directly
            model = deployed
            try:
                model_xpu = xpu_device.XPU.coerce(model)
                if xpu != model_xpu:
                    log('Re-Mount model on {}'.format(xpu))
                    model = xpu.mount(model)
            except Exception:
                log('Re-Mount model on {}'.format(xpu))
                model = xpu.mount(model)
        elif isinstance(deployed, DeployedModel):
            model = deployed.load_model()
            log('Mount {} on {}'.format(deployed, xpu))
            model = xpu.mount(model)
        else:
            raise TypeError('Unable to ensure {!r} as a mounted model'.format(
                deployed))

        return model, xpu

    @classmethod
    def coerce(DeployedModel, arg):
        """
        Attempt to coerce the argument into a deployed model.

        Args:
            arg (DeployedModel | PathLike | torch.nn.Module) : can be:
                (1) a DeployedModel object
                (2) a path to a deploy file
                (3) a live pytorch module
                (4) a path to a .pt file in a train snapshot directory.

        Returns:
            DeployedModel
        """
        from os.path import dirname
        import torch
        if isinstance(arg, DeployedModel):
            # The input is already a DeployedModel
            deployed = arg
        elif isinstance(arg, torch.nn.Module):
            # The argument is a live pytorch model
            deployed = DeployedModel(None)
            deployed._model = arg
        elif isinstance(arg, str):
            # handle the case where we are given a weights file
            # use heuristics try and determine model topology
            if arg.endswith('.pt'):
                snap_fpath = arg
                dpath_cands = []
                # Look the pt file's directory for topology and train info
                dpath1 = dirname(snap_fpath)
                dpath_cands = [dpath1]
                # The files might also be in the parent directory
                if not exists(join(dpath1, 'train_info.json')):
                    dpath_cands.append(dirname(dpath1))
                # Search for the files in the candidate directories
                train_info_cands = list(ub.find_path(
                    'train_info.json', path=dpath_cands, exact=True))
                model_cands = list(ub.find_path(
                    '*.py', path=dpath_cands, exact=False))
                if len(model_cands) == 0:
                    raise Exception('Model topology does not exist for {!r}.'.format(arg))
                elif len(model_cands) > 1:
                    raise Exception('Conflicting model topologies for {!r}.'.format(arg))
                else:
                    model_fpath = model_cands[0]
                if len(train_info_cands) == 0:
                    train_info_fpath = None
                elif len(train_info_cands) > 1:
                    raise AssertionError('Conflicting train_info.json files')
                else:
                    train_info_fpath = train_info_cands[0]
                deployed = DeployedModel.custom(snap_fpath, model_fpath,
                                                train_info_fpath=train_info_fpath)
            else:
                # Assume we have a deploy path
                deployed = DeployedModel(arg)
        else:
            # Unhandled case
            raise TypeError(type(arg))
        return deployed

    @classmethod
    def custom(DeployedModel, snap_fpath, model, initkw=None, train_info_fpath=None):
        """
        Create a deployed model even if the model wasnt trained with FitHarn

        This just requires specifying a bit more information, which FitHarn
        would have tracked.

        Args:
            snap_fpath (PathLike):
                path to the exported (snapshot) weights file

            model (PathLike or nn.Module): can either be
                (1) a path to model topology (created via `export_model_code`)
                (2) the model class or an instance of the class

            initkw (Dict): if model is a class or instance, then
                you must pass the keyword arguments used to construct it.

            train_info_fpath (PathLike, optional):
                path to a json file containing additional training metadata

        Example:
            >>> # Setup raw components
            >>> # xdoctest: +REQUIRES(module:netharn)
            >>> train_dpath = _demodata_trained_dpath()
            >>> deployed = DeployedModel(train_dpath)
            >>> snap_fpath = deployed.info['snap_fpath']
            >>> model, initkw = deployed.model_definition()
            >>> train_info_fpath = deployed.info['train_info_fpath']
            >>> # Past raw components to custom
            >>> self = DeployedModel.custom(snap_fpath, model, initkw)
            >>> dpath = ub.ensure_app_cache_dir('torch_liberator', 'tests/_package_custom')
            >>> self.package(dpath)
        """
        if isinstance(model, str):
            model_fpath = model
            if initkw is not None:
                raise ValueError('initkw not used when model is a path')
        else:
            import tempfile
            from . import exporter
            dpath = tempfile.mkdtemp()
            model_fpath = exporter.export_model_code(dpath, model, initkw=initkw)

        _info = {
            'model_fpath': model_fpath,
            'snap_fpath': snap_fpath,
            'train_info_fpath': train_info_fpath,
        }
        self = DeployedModel(None)
        self._info = _info
        return self


def _demodata_zip_fpath():
    zip_path = DeployedModel(_demodata_trained_dpath()).package()
    return zip_path


def _demodata_toy_harn():
    # TODO: replace with lightning
    # This will train a toy model with toy data using netharn
    import netharn as nh
    hyper = nh.HyperParams(**{
        'workdir'     : ub.ensure_app_cache_dir('torch_liberator/tests/deploy'),
        'name'        : 'demo_liberator_static',
        'xpu'         : nh.XPU.coerce('cpu'),
        'datasets'    : {'train': nh.data.ToyData2d(size=3, rng=0)},
        'loaders'     : {'batch_size': 64},
        'model'       : (nh.models.ToyNet2d, {}),
        'optimizer'   : (nh.optimizers.SGD, {'lr': 0.0001}),
        'criterion'   : (nh.criterions.FocalLoss, {}),
        'initializer' : (nh.initializers.KaimingNormal, {}),
        'monitor'     : (nh.Monitor, {'max_epoch': 1}),
    })
    harn = nh.FitHarn(hyper)
    harn.preferences['use_tensorboard'] = False
    harn.preferences['log_gradients'] = False
    # harn.preferences['timeout'] = 30
    return harn


def _demodata_trained_dpath():
    harn = _demodata_toy_harn()
    harn.run()  # TODO: make this run faster if we don't need to rerun
    if len(list(glob.glob(join(harn.train_dpath, '*.py')))) > 1:
        # If multiple models are deployed some hash changed. Need to reset
        harn.initialize(reset='delete')
        harn.run()  # don't relearn if we already finished this one
    return harn.train_dpath


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m torch_liberator.deployer all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
