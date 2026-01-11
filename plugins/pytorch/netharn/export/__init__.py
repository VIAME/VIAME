r"""
NOTICE: ``netharn.export`` has been refactored into the packages ``liberator``
which performs general code extraction and ``torch_liberator`` which is
specific to pytorch. This module is deprecated and will be removed in the
future.

For now here are old docs, slightly updated to reference the correct packages:

The package torch_liberator.deployed contains DeployedModel, which consists of
logic to take the model topology definition along with the "best" snapshot in a
training directory and package it up into a standalone zipfile. The
DeployedModel can also be used to reload model from this zipfile. Thus this
zipfile can be passed around as a pytorch model topology+pretrained weights
transfer format.

The file torch_liberator.exporter contains the code that simply exports the
model toplogy via code Uses static analysis to export relevant code that
defines the model topology into a stanadlone file. As long as your model
definition is indepenent of your training code, then the exported file can be
passed around in a similar way to a caffe prototext file.


CommandLine:
    # Runs the following example
    xdoctest -m netharn.export __doc__:0

Example:
    >>> # xdoctest: +IGNORE_WANT
    >>> # This example will train a small model and then deploy it.
    >>> import netharn as nh
    >>> import ubelt as ub
    >>> #
    >>> #################################################
    >>> print('--- STEP 1: TRAIN A MODEL ---')
    >>> # This will train a toy model with toy data using netharn
    >>> hyper = nh.HyperParams(**{
    >>>     'workdir'     : ub.ensure_app_cache_dir('netharn/tests/deploy'),
    >>>     'name'        : 'deploy_demo',
    >>>     'xpu'         : nh.XPU.coerce('cpu'),
    >>>     'datasets'    : {
    >>>         'train': nh.data.ToyData2d(size=3, border=1, n=256, rng=0),
    >>>         'test':  nh.data.ToyData2d(size=3, border=1, n=128, rng=1),
    >>>     },
    >>>     'loaders'     : {'batch_size': 64},
    >>>     'model'       : (nh.models.ToyNet2d, {}),
    >>>     'optimizer'   : (nh.optimizers.SGD, {'lr': 0.0001}),
    >>>     'criterion'   : (nh.criterions.CrossEntropyLoss, {}),
    >>>     'initializer' : (nh.initializers.KaimingNormal, {}),
    >>>     'scheduler'   : (nh.schedulers.ListedLR, {
    >>>         'points': {0: .01, 3: 0.1},
    >>>         'interpolate': True,
    >>>     }),
    >>>     'monitor'     : (nh.Monitor, {'max_epoch': 3,}),
    >>> })
    >>> harn = nh.FitHarn(hyper)
    >>> harn.preferences['use_tensorboard'] = False
    >>> harn.preferences['timeout'] = 10
    >>> harn.intervals['test'] = 1
    >>> harn.initialize(reset='delete')
    >>> harn.run()
    --- STEP 1: TRAIN A MODEL ---
    RESET HARNESS BY DELETING EVERYTHING IN TRAINING DIR
    Symlink: .../.cache/netharn/tests/deploy/fit/runs/deploy_demo/onnxqaww -> .../.cache/netharn/tests/deploy/_mru
    ......
    Symlink: .../.cache/netharn/tests/deploy/fit/runs/deploy_demo/onnxqaww -> .../.cache/netharn/tests/deploy/fit/nice/deploy_demo
    ......
    INFO: Model has 824 parameters
    INFO: Mounting ToyNet2d model on CPU
    INFO: Exported model topology to .../.cache/netharn/tests/deploy/fit/runs/deploy_demo/onnxqaww/ToyNet2d_2a3f49.py
    INFO: Initializing model weights with: <netharn.initializers.nninit_core.KaimingNormal object at 0x7fc67efdf8d0>
    INFO:  * harn.train_dpath = '.../.cache/netharn/tests/deploy/fit/runs/deploy_demo/onnxqaww'
    INFO:  * harn.name_dpath  = '.../.cache/netharn/tests/deploy/fit/name/deploy_demo'
    INFO: Snapshots will save to harn.snapshot_dpath = '.../.cache/netharn/tests/deploy/fit/runs/deploy_demo/onnxqaww/torch_snapshots'
    INFO: ARGV:
        .../.local/conda/envs/py36/bin/python .../.local/conda/envs/py36/bin/ipython
    INFO: === begin training 0 / 3 : deploy_demo ===
    epoch lr:0.01 │ vloss is unevaluated 0/3... rate=0 Hz, eta=?, total=0:00:00, wall=19:32 EST
    train loss:0.717 │ 100.00% of 64x8... rate=2093.02 Hz, eta=0:00:00, total=0:00:00, wall=19:32 EST
    test loss:0.674 │ 100.00% of 64x4... rate=14103.48 Hz, eta=0:00:00, total=0:00:00, wall=19:32 EST
    Populating the interactive namespace from numpy and matplotlib
    INFO: === finish epoch 0 / 3 : deploy_demo ===
    epoch lr:0.04 │ vloss is unevaluated 1/3... rate=0.87 Hz, eta=0:00:02, total=0:00:01, wall=19:32 EST
    train loss:0.712 │ 100.00% of 64x8... rate=2771.29 Hz, eta=0:00:00, total=0:00:00, wall=19:32 EST
    test loss:0.663 │ 100.00% of 64x4... rate=15867.59 Hz, eta=0:00:00, total=0:00:00, wall=19:32 EST
    INFO: === finish epoch 1 / 3 : deploy_demo ===
    epoch lr:0.07 │ vloss is unevaluated 2/3... rate=1.04 Hz, eta=0:00:00, total=0:00:01, wall=19:32 EST
    train loss:0.686 │ 100.00% of 64x8... rate=2743.56 Hz, eta=0:00:00, total=0:00:00, wall=19:32 EST
    test loss:0.636 │ 100.00% of 64x4... rate=14332.63 Hz, eta=0:00:00, total=0:00:00, wall=19:32 EST
    INFO: === finish epoch 2 / 3 : deploy_demo ===
    epoch lr:0.1 │ vloss is unevaluated 3/3... rate=1.11 Hz, eta=0:00:00, total=0:00:02, wall=19:32 EST
    INFO: Maximum harn.epoch reached, terminating ...
    INFO:
    INFO: training completed
    INFO: harn.train_dpath = '.../.cache/netharn/tests/deploy/fit/runs/deploy_demo/onnxqaww'
    INFO: harn.name_dpath  = '.../.cache/netharn/tests/deploy/fit/name/deploy_demo'
    INFO: view tensorboard results for this run via:
        tensorboard --logdir ~/.cache/netharn/tests/deploy/fit/name
    [DEPLOYER] Deployed zipfpath=.../.cache/netharn/tests/deploy/fit/runs/deploy_demo/onnxqaww/deploy_ToyNet2d_onnxqaww_002_TXZBYL.zip
    INFO: wrote single-file deployment to: '.../.cache/netharn/tests/deploy/fit/runs/deploy_demo/onnxqaww/deploy_ToyNet2d_onnxqaww_002_TXZBYL.zip'
    INFO: exiting fit harness.
    Out[1]: '.../.cache/netharn/tests/deploy/fit/runs/deploy_demo/onnxqaww/deploy_ToyNet2d_onnxqaww_002_TXZBYL.zip'
    >>> #
    >>> ##########################################
    >>> print('--- STEP 2: DEPLOY THE MODEL ---')
    >>> # First we export the model topology to a standalone file
    >>> # (Note: this step is done automatically in `harn.run`, but we do
    >>> #  it again here for demo purposes)
    >>> import torch_liberator
    >>> topo_fpath = torch_liberator.export_model_code(harn.train_dpath, harn.hyper.model_cls, harn.hyper.model_params)
    >>> # Now create an instance of deployed model that points to the
    >>> # Training dpath. (Note the directory structure setup by netharn is
    >>> # itself a deployment, it just has multiple files)
    >>> import time
    >>> deployer = torch_liberator.DeployedModel(harn.train_dpath)
    >>> train_path = ub.Path(harn.train_dpath)
    >>> print(ub.repr2(list(train_path.walk())))
    >>> print('deployer.info = {}'.format(ub.repr2(deployer.info, nl=1)))
    >>> # Use the DeployedModel to package the imporant info in train_dpath
    >>> # into a standalone zipfile.
    >>> zip_fpath = deployer.package()
    >>> print('We exported the topology to: {!r}'.format(topo_fpath))
    >>> print('We exported the topology+weights to: {!r}'.format(zip_fpath))
    --- STEP 2: DEPLOY THE MODEL ---
    We exported the topology to: '...tests/deploy/fit/runs/deploy_demo/onnxqaww/ToyNet2d_2a3f49.py'
    We exported the topology+weights to: '...tests/deploy/fit/runs/deploy_demo/onnxqaww/deploy_ToyNet2d_onnxqaww_002_HVWCGI.zip'
    >>> #
    >>> #################################################
    >>> print('--- STEP 3: LOAD THE DEPLOYED MODEL ---')
    >>> # Now we can move the zipfile anywhere we want, and we should
    >>> # still be able to load it (depending on how coupled the model is).
    >>> # Create an instance of DeployedModel that points to the zipfile
    >>> # (Note: DeployedModel is used to both package and load models)
    >>> loader = torch_liberator.DeployedModel(zip_fpath)
    >>> model = loader.load_model()
    >>> # This model is now loaded with the corret weights.
    >>> # You can use it as normal.
    >>> model.eval()
    >>> images = harn._demo_batch(0)[0][0:1]
    >>> outputs = model(images)
    >>> print('outputs = {!r}'.format(outputs))
    >>> # Not that the loaded model is independent of harn.model
    >>> print('model.__module__ = {!r}'.format(model.__module__))
    >>> print('harn.model.module.__module__ = {!r}'.format(harn.model.module.__module__))
    --- STEP 3: LOAD THE DEPLOYED MODEL ---
    outputs = tensor([[0.4105, 0.5895]], grad_fn=<SoftmaxBackward>)
    model.__module__ = 'deploy_ToyNet2d_onnxqaww_002_HVWCGI/ToyNet2d_2a3f49'
    harn.model.module.__module__ = 'netharn.models.toynet'
    >>> model = None
    >>> loader = None
    >>> outputs = None
    >>> images = None
"""
from netharn.export import deployer
from netharn.export import exporter

from netharn.export.deployer import (DeployedModel,)
from netharn.export.exporter import (export_model_code,)
import warnings
warnings.warn('netharn.export is deprecated, use torch_liberator intead', DeprecationWarning)

__all__ = ['DeployedModel', 'deployer', 'export_model_code', 'exporter']
