"""
Logic for detecting on a truth dataset using a pretrained network, and then
scoring those detections.

TODO:
    - [ ] create CLI flag to reduce dataset size for debugging
         * Note: if we determine an optimal budget for test data size, then we
         have the option to use reintroduce rest back into the training set.
"""
from os.path import exists
from os.path import join
from os.path import dirname
import os
import six
import kwcoco
import ndsampler
import kwimage
from viame.pytorch import netharn as nh
import numpy as np
import ubelt as ub
import scriptconfig as scfg
from viame.pytorch.netharn import detect_predict
from viame.pytorch.netharn.data import data_containers  # NOQA
from viame.pytorch.netharn.data import channel_spec  # NOQA


class DetectEvaluateConfig(scfg.Config):
    default = {

        'deployed': scfg.Value(None, nargs='+', help='deployed network filepath'),

        # Evaluation dataset
        'dataset': scfg.Value(None, help='path to an mscoco dataset'),
        'workdir': scfg.Path('~/work/bioharn', help='Workdir for sampler'),

        'batch_size': scfg.Value(4, help=(
            'number of images that run through the network at a time')),

        'input_dims': scfg.Value('native', help=(
            'size of input chip; or "native" which uses input_dims used in training')),

        'window_dims': scfg.Value('native', help=(
            'size of sliding window chip; or "full" which uses the entire frame; '
            'or "native", which uses window_dims specified in training')),

        'window_overlap': scfg.Value(0.0, help='overlap of the sliding window'),

        'sampler_backend': scfg.Value(None, help='ndsampler backend'),

        'workers': scfg.Value(4, help='num CPUs for data loading'),

        'verbose': 1,

        # Note: these dont work exactly correct due to mmdetection model
        # differences
        'nms_thresh': 0.4,
        'conf_thresh': 0.1,

        'xpu': scfg.Value('auto', help='a CUDA device or a CPU'),

        'channels': scfg.Value(
            'native', type=str,
            help='a specification of channels needed by this model. See ChannelSpec for details. '
            'Typically this can be inferred from the model'),

        # 'out_dpath': scfg.Path('./detect_eval_out/', help='folder to send the output'),
        'out_dpath': scfg.Path('special:train_dpath', help='folder to send the output'),

        'eval_in_train_dpath': scfg.Path(True, help='write eval results into the training directory if its known'),

        'draw': scfg.Value(10, help='number of images with predictions to draw'),
        'enable_cache': scfg.Value(True, help='writes predictions to disk'),

        'demo': scfg.Value(False, help='debug helper'),

        'classes_of_interest': scfg.Value([], help='if specified only these classes are given weight'),

        'async_buffer': scfg.Value(False, help="I've seen this increase prediction rate but it also increases instability, unsure of the reason"),

        # kwcoco eval config overrides
        'compat': scfg.Value(
            value='all',
            choices=['all', 'mutex', 'ancestors'],
            help=ub.paragraph(
                '''
                Matching strategy for which true annots are allowed to match
                which predicted annots.
                `mutex` means true boxes can only match predictions where the
                true class has highest probability (pycocotools setting).
                `all` means any class can match any other class.
                Dont use `ancestors`, it is broken.
                ''')),

        'monotonic_ppv': scfg.Value(False, help=ub.paragraph(
            '''
            if True forces precision to be monotonic. Defaults to True for
            compatibility with pycocotools, but that might not be the best
            option.
            ''')),

        'ap_method': scfg.Value('sklearn', help=ub.paragraph(
            '''
            Method for computing AP. Defaults to a setting comparable to
            pycocotools. Can also be set to sklearn to use an alterative
            method.
            ''')),

        'area_range': scfg.Value(
            value=['all'],
            # value='0-inf,0-32,32-96,96-inf',
            help=(
                'minimum and maximum object areas to consider. '
                'may be specified as a comma-separated code: <min>-<max>. '
                'also accepts keys all, small, medium, and large. '
            )),

        'iou_bias': scfg.Value(0, help=(
            'pycocotools setting is 1, but 0 may be better')),

        'skip_upgrade': scfg.Value(False, help='if true skips upgrade model checks'),
    }


def evaluate_models(cmdline=True, **kw):
    """
    Evaluate multiple models using a config file or CLI.

    /home/joncrall/work/bioharn/fit/nice/bioharn-det-v16-cascade/deploy_MM_CascadeRCNN_hvayxfyx_036_TLRPCP.zip

    Ignore:
        from .detect_eval import *  # NOQA
        kw = {}

        >>> from .detect_eval import *  # NOQA
        >>> config = DetectEvaluator.demo_config()
        >>> kw = dict(config)
        >>> deployed = kw.pop('deployed')
        >>> kw['deployed'] = [deployed, deployed]
        >>> evaluate_models(**kw)

        kw = {
            'deployed': '~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/manual-snapshots/_epoch_00000006.pt',
            'workers': 4,
            'batch_size': 10,
            'xpu': 0,
        }
        evaluate_models(**kw)
    """
    import itertools as it
    from kwcoco.util.util_json import ensure_json_serializable
    import json
    import pandas as pd
    if 'config' in kw:
        config_fpath = kw['config']
        defaults = ub.dict_diff(kw, {'config'})
        multi_config = DetectEvaluateConfig(data=config_fpath, default=defaults, cmdline=cmdline)
    else:
        multi_config = DetectEvaluateConfig(default=kw, cmdline=cmdline)
    print('MultiConfig: {}'.format(ub.repr2(multi_config.asdict())))

    # Look for specific items in the base config where multiple values are
    # given. We will evaluate over all permutations of these values.
    base_config = multi_config.asdict()

    model_fpaths = base_config.pop('deployed')
    if not ub.iterable(model_fpaths):
        model_fpaths = [model_fpaths]
    model_fpaths = [ub.expandpath(p) for p in model_fpaths]

    for fpath in model_fpaths:
        if not exists(fpath):
            raise Exception('{} does not exist'.format(fpath))

    search_space = {}
    input_dims = base_config.pop('input_dims')
    if ub.iterable(input_dims) and len(input_dims) and ub.iterable(input_dims[0], strok=True):
        # multiple input dims to test are given as a list
        search_space['input_dims'] = input_dims
    else:
        # only one input given, need to wrap
        search_space['input_dims'] = [input_dims]

    keys = list(search_space.keys())
    basis = list(search_space.values())
    config_perms = [ub.dzip(keys, permval) for permval in it.product(*basis)]

    sampler = None
    metric_fpaths = []
    for model_fpath in ub.ProgIter(model_fpaths, desc='test model', verbose=3):
        print('model_fpath = {!r}'.format(model_fpath))

        predictor = None
        for perm in ub.ProgIter(config_perms, desc='test config', verbose=3):

            # Create the config for this detection permutation
            config = ub.dict_union(base_config, perm)
            config['deployed'] = model_fpath

            print('config = {}'.format(ub.repr2(config)))
            evaluator = DetectEvaluator(config)

            # Reuse the dataset / predictor when possible
            evaluator.predictor = predictor
            evaluator.sampler = sampler
            print('_init')
            evaluator._init()
            print('evaluate')

            results = evaluator.evaluate()

            single_result = results['area_range=all,iou_thresh=0.5']
            small_results = {
                'nocls_measures': single_result.nocls_measures.summary(),
                'ovr_measures': single_result.ovr_measures.summary(),
                'meta': single_result.meta,
            }
            small_results = ensure_json_serializable(small_results, normalize_containers=True)
            small_metrics_fpath = join(evaluator.paths['metrics'], 'small_metrics.json')
            print('small_metrics_fpath = {!r}'.format(small_metrics_fpath))
            with open(small_metrics_fpath, 'w') as file:
                json.dump(small_results, file, indent='    ')
            metric_fpaths.append(small_metrics_fpath)

            # Save loaded predictor/sampler for the next run of this model/dataset
            predictor = evaluator.predictor
            sampler = evaluator.sampler

    rows = []
    train_config_rows = []
    # import ast
    for fpath in ub.ProgIter(metric_fpaths, desc='gather summary'):
        metrics = json.load(open(fpath, 'r'))
        row = {}
        row['model_tag'] = metrics['meta']['model_tag']
        row['predcfg_tag'] = metrics['meta']['predcfg_tag']
        row['ap'] = metrics['nocls_measures']['ap']
        row['auc'] = metrics['nocls_measures']['auc']

        # Hack to get train config params
        # train_config = ast.literal_eval(metrics['train_info']['extra']['config'])
        train_config = eval(metrics['meta']['train_info']['extra']['config'],
                            {'inf': float('inf')}, {})
        train_config_rows.append(train_config)
        rows.append(row)

    pd.set_option('max_colwidth', 256)
    df = pd.DataFrame(rows)
    print(df.to_string(float_format=lambda x: '%0.3f' % x))

    def find_varied_params(train_config_rows):
        all_keys = set()
        for c in train_config_rows:
            all_keys.update(set(c))
        ignore_keys = {
            'datasets', 'focus', 'max_epoch', 'nice', 'ovthresh', 'patience',
            'workdir', 'workers', 'xpu', 'sampler_backend', 'visible_thresh',
            'warmup_iters', 'pretrained', 'grad_norm_type', 'grad_norm_max',
            'warmup_ratio',
        }
        valid_keys = all_keys - ignore_keys
        key_basis = ub.ddict(set)
        for c in train_config_rows:
            for k in valid_keys:
                v = c.get(k, ub.NoParam)
                if isinstance(v, list):
                    v = tuple(v)
                key_basis[k].add(v)
        varied_basis = {}

        force_include_keys = {
            'window_overlap',
            'batch_size', 'augment', 'init',  'bstep',
            'input_dims', 'lr', 'channels',  'multiscale',
            'normalize_inputs', 'window_dims',
        }
        for k, vs in list(key_basis.items()):
            if len(vs) > 1 or k in force_include_keys:
                varied_basis[k] = set(vs)
        return varied_basis

    varied_basis = find_varied_params(train_config_rows)

    for row, config in zip(rows, train_config_rows):
        subcfg = ub.dict_subset(config, set(varied_basis), default=np.nan)
        row.update(subcfg)

    pd.set_option('max_colwidth', 256)
    df = pd.DataFrame(rows)
    print(df.to_string(float_format=lambda x: '%0.3f' % x))


class DetectEvaluator(object):
    """
    Evaluation harness for a detection task.

    Creates an instance of :class:`bioharn.detect_predict.DetectPredictor`,
    executes prediction, compares the results to a groundtruth dataset, and
    outputs various metrics summarizing performance.

    Args:
        config (DetectEvaluateConfig):
            the configuration of the evaluator, which is a superset of
            :class:`bioharn.detect_predict.DetectPredictConfig`.

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> from .detect_eval import *  # NOQA
        >>> # See DetectEvaluateConfig for config docs
        >>> config = DetectEvaluator.demo_config()
        >>> evaluator = DetectEvaluator(config)
        >>> evaluator.evaluate()

    Ignore:
        from .detect_eval import *  # NOQA
        config = {}
        config['deployed'] = ub.expandpath('$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-v30-bigger-balanced/etvvhzni/deploy_MM_CascadeRCNN_etvvhzni_007_IPEIQA.zip')
        config['dataset'] = ub.expandpath('$HOME/remote/namek/data/private/_combos/test_cfarm_habcam_v1.mscoco.json')

        evaluator = DetectEvaluator(config)
        evaluator._init()
        predictor = evaluator.predictor
        sampler = evaluator.sampler
        coco_dset = sampler.dset

        predictor.config['verbose'] = 1
        out_dpath = evaluator.paths['base']
        evaluator.evaluate()
    """

    def __init__(evaluator, config=None):
        evaluator.config = DetectEvaluateConfig(config)
        evaluator.predictor = None
        evaluator.sampler = None

    @classmethod
    def demo_config(cls):
        """
        Train a small demo model

        Example:
            >>> from .detect_eval import *  # NOQA
            >>> config = DetectEvaluator.demo_config()
            >>> print('config = {}'.format(ub.repr2(config, nl=1)))
        """
        from viame.pytorch.netharn import detect_fit
        aux = False

        train_dset = kwcoco.CocoDataset.demo('shapes8', aux=aux)
        dpath = ub.ensure_app_cache_dir('bioharn/demodata')
        test_dset = kwcoco.CocoDataset.demo('shapes4', aux=aux)
        workdir = ub.ensuredir((dpath, 'work'))

        train_dset.fpath = join(dpath, 'shapes_train.mscoco')
        train_dset.dump(train_dset.fpath)

        test_dset.fpath = join(dpath, 'shapes_test.mscoco')
        test_dset.dump(test_dset.fpath)
        channels = 'rgb|disparity' if aux else 'rgb'

        deploy_fpath = detect_fit.fit(
            # arch='cascade',
            arch='yolo2',
            train_dataset=train_dset.fpath,
            channels=channels,
            workers=0,
            workdir=workdir,
            batch_size=2,
            window_dims=(256, 256),
            max_epoch=2,
            timeout=60,
            # timeout=1,
        )

        train_dpath = dirname(deploy_fpath)
        out_dpath = ub.ensuredir(train_dpath, 'out_eval')

        config = {
            'deployed': deploy_fpath,

            'dataset': test_dset.fpath,
            'workdir': workdir,
            'out_dpath': out_dpath,
        }
        return config

    def _init(evaluator):
        evaluator._ensure_sampler()
        evaluator._init_predictor()

    def _ensure_sampler(evaluator):
        if evaluator.sampler is None:
            print('loading dataset')
            coco_dset = kwcoco.CocoDataset.coerce(evaluator.config['dataset'])

            if evaluator.config['demo']:
                pass
            print('loaded dataset')
            workdir = ub.expandpath(evaluator.config['workdir'])
            sampler = ndsampler.CocoSampler(coco_dset, workdir=workdir,
                                            backend=evaluator.config['sampler_backend'])
            evaluator.sampler = sampler
            # evaluator.sampler.frames.prepare(workers=min(2, evaluator.config['workers']))
            print('prepare frames')
            evaluator.sampler.frames.prepare(workers=evaluator.config['workers'])
            print('finished dataset load')

    def _init_predictor(evaluator):
        # Load model
        import torch_liberator
        deployed = torch_liberator.DeployedModel.coerce(evaluator.config['deployed'])
        if hasattr(deployed, '_train_info'):
            evaluator.train_info = deployed._train_info
        else:
            evaluator.train_info = deployed.train_info()
        nice = evaluator.train_info['nice']

        # hack together a model tag
        if hasattr(deployed, 'model_tag'):
            model_tag = deployed.model_tag
        else:
            if deployed.path is None:
                model_tag = nice + '_' + ub.augpath(deployed._info['snap_fpath'], dpath='', ext='', multidot=True)
            else:
                model_tag = nice + '_' + ub.augpath(deployed.path, dpath='', ext='', multidot=True)

        def removesuffix(self, suffix):
            """ 3.9 backport https://www.python.org/dev/peps/pep-0616/ """
            if suffix and self.endswith(suffix):
                return self[:-len(suffix)]
            else:
                return self[:]

        evaluator.model_tag = model_tag
        evaluator.dset_tag = removesuffix(evaluator.sampler.dset.tag, '.json')

        # Load the trained model
        pred_keys = set(detect_predict.DetectPredictConfig.default.keys()) - {'verbose'}
        pred_cfg = ub.dict_subset(evaluator.config, pred_keys)

        # if evaluator.config['input_dims'] == 'native':
        #     # hack, this info exists, but not in an easy form
        #     train_config = eval(deployed.train_info()['extra']['config'], {})
        #     pred_cfg['input_dims'] = train_config['input_dims']

        native = detect_predict.DetectPredictor._infer_native(pred_cfg)
        pred_cfg.update(native)

        if evaluator.predictor is None:
            # Only create the predictor if needed
            print('Needs initial init')
            evaluator.predictor = detect_predict.DetectPredictor(pred_cfg)
            evaluator.predictor._ensure_model()
        else:
            # Reuse loaded predictors from other evaluators.
            # Update the config in this case
            needs_reinit = evaluator.predictor.config['deployed'] != pred_cfg['deployed']
            evaluator.predictor.config.update(pred_cfg)
            print('needs_reinit = {!r}'.format(needs_reinit))
            if needs_reinit:
                evaluator.predictor._ensure_model()
            else:
                print('reusing loaded model')

        evaluator.classes = evaluator.predictor.raw_model.classes

        # The parameters that influence the predictions
        pred_params = ub.dict_subset(evaluator.predictor.config, [
            'input_dims',
            'window_dims',
            'window_overlap',
            'conf_thresh',
            'nms_thresh',
        ])
        evaluator.pred_cfg = nh.util.make_short_idstr(pred_params)
        evaluator.predcfg_tag = evaluator.pred_cfg

        # ---- PATHS ----

        # TODO: make path initialization separate?
        # evaluator._init_paths()
        # def _init_paths(evaluator):

        require_train_dpath = evaluator.config['eval_in_train_dpath']
        out_dpath = evaluator.config['out_dpath']
        out_dpath = None
        if isinstance(out_dpath, six.string_types):
            if out_dpath == 'special:train_dpath':
                out_dpath = None
                require_train_dpath = True
            else:
                out_dpath = ub.ensuredir(evaluator.config['out_dpath'])

        # Use tags to make a relative directory structure based on configs
        rel_cfg_dir = join(evaluator.dset_tag, evaluator.model_tag,
                           evaluator.pred_cfg)

        class UnknownTrainDpath(Exception):
            pass

        def _introspect_train_dpath(deployed):
            # NOTE: the train_dpath in the info directory is wrt to the
            # machine the model was trained on. Used the deployed model to
            # grab that path instead wrt to the current machine.
            if hasattr(deployed, 'train_dpath'):
                train_dpath = deployed.train_dpath
            else:
                if deployed.path is None:
                    train_dpath = dirname(deployed.info['train_info_fpath'])
                else:
                    if os.path.isdir(deployed.path):
                        train_dpath = deployed.path
                    else:
                        train_dpath = dirname(deployed.path)
            print('train_dpath = {!r}'.format(train_dpath))
            return train_dpath

        try:
            if require_train_dpath:
                train_dpath = _introspect_train_dpath(deployed)
                assert exists(train_dpath), (
                    'train_dpath={} does not exist. Is this the right '
                    'machine?'.format(train_dpath))
                eval_dpath = join(train_dpath, 'eval', rel_cfg_dir)
                ub.ensuredir(eval_dpath)

                if out_dpath is not None:
                    base_dpath = join(out_dpath, rel_cfg_dir)
                    ub.ensuredir(dirname(base_dpath))
                    if not os.path.islink(base_dpath) and exists(base_dpath):
                        ub.delete(base_dpath)
                    ub.symlink(eval_dpath, base_dpath, overwrite=True, verbose=3)
                else:
                    base_dpath = eval_dpath
            else:
                raise UnknownTrainDpath
        except UnknownTrainDpath:
            if out_dpath is None:
                raise Exception('Must specify out_dpath if train_dpath is unknown')
            else:
                base_dpath = join(out_dpath, rel_cfg_dir)
                ub.ensuredir(base_dpath)

        evaluator.paths = {}
        evaluator.paths['base'] = base_dpath
        evaluator.paths['metrics'] = ub.ensuredir((evaluator.paths['base'], 'metrics'))
        evaluator.paths['viz'] = ub.ensuredir((evaluator.paths['base'], 'viz'))
        print('evaluator.paths = {}'.format(ub.repr2(evaluator.paths, nl=1)))

    def _run_predictions(evaluator):

        predictor = evaluator.predictor
        sampler = evaluator.sampler
        # pred_gen = evaluator.predictor.predict_sampler(sampler)

        predictor.config['verbose'] = 1

        out_dpath = evaluator.paths['base']

        gids = None
        # gids = sorted(sampler.dset.imgs.keys())[0:10]

        draw = evaluator.config['draw']
        enable_cache = evaluator.config['enable_cache']
        async_buffer = evaluator.config['async_buffer']

        gid_to_pred, gid_to_pred_fpath = detect_predict._cached_predict(
            predictor, sampler, out_dpath, gids=gids,
            draw=draw,
            async_buffer=async_buffer,
            enable_cache=enable_cache,
        )
        return gid_to_pred

    def evaluate(evaluator):
        """
        Ignore:
            >>> config = dict(
            >>>     dataset=ub.expandpath('$HOME/data/noaa_habcam/combos/habcam_cfarm_v6_test.mscoco.json'),
            >>>     deployed=ub.expandpath('$HOME/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-v36/brekugqz/_epoch_00000018.pt'),
            >>>     sampler_backend='cog', batch_size=256,
            >>>     conf_thresh=0.2, nms_thresh=0.5
            >>> )
            >>> evaluator = DetectEvaluator(config)
            >>> evaluator.evaluate()
        """
        if evaluator.predictor is None or evaluator.sampler is None:
            evaluator._init()

        evaluator.predictor.config['verbose'] = 3
        gid_to_pred = evaluator._run_predictions()

        truth_sampler = evaluator.sampler

        # TODO: decouple this (predictor + evaluator) with CocoEvaluator (evaluator)
        classes_of_interest = evaluator.config['classes_of_interest']

        if 0:
            classes_of_interest = [
                'flatfish', 'live sea scallop', 'dead sea scallop']

        ignore_classes = {'ignore'}
        # ignore_class_freq_thresh = 200
        # truth_sampler = evaluator.sampler
        # true_catfreq = truth_sampler.dset.category_annotation_frequency()
        # rare_canames = {cname for cname, freq in true_catfreq.items()
        #                 if freq < ignore_class_freq_thresh}
        # ignore_classes.update(rare_canames)

        expt_title = '{} {}\n{}'.format(
            evaluator.model_tag, evaluator.predcfg_tag, evaluator.dset_tag,)

        metrics_dpath = evaluator.paths['metrics']

        # TODO: clean-up decoupling
        from kwcoco import coco_evaluator
        coco_eval_config = ub.dict_isect(
            evaluator.config, coco_evaluator.CocoEvalConfig.default)
        coco_eval_config.update({
            'true_dataset': truth_sampler,
            'classes_of_interest': classes_of_interest,
            'ignore_classes': ignore_classes,
            # 'out_dpath': metrics_dpath,
            # 'expt_title': expt_title,
            # 'draw': False,  # hack while this still exists
        })
        print('coco_eval_config = {}'.format(ub.repr2(coco_eval_config, nl=1)))
        coco_eval_config['pred_dataset'] = gid_to_pred
        coco_eval = coco_evaluator.CocoEvaluator(coco_eval_config)
        coco_eval._init()
        results = coco_eval.evaluate()

        eval_config = evaluator.config.asdict()

        from kwcoco.util.util_json import ensure_json_serializable
        eval_config = ensure_json_serializable(
            eval_config, normalize_containers=True)

        extra_meta = {
            'dset_tag': evaluator.dset_tag,
            'model_tag': evaluator.model_tag,
            'predcfg_tag': evaluator.predcfg_tag,

            'ignore_classes': sorted(ignore_classes),

            'eval_config': eval_config,
            'train_info': evaluator.train_info,
        }
        # results.meta.update(extra_meta)
        for subres in results.values():
            subres.meta.update(extra_meta)

        metrics_fpath = join(metrics_dpath, 'metrics.json')
        print('dumping metrics_fpath = {!r}'.format(metrics_fpath))
        results.dump(metrics_fpath, indent='    ')
        results.dump_figures(
            out_dpath=metrics_dpath,
            expt_title=expt_title)

        if True:
            print('Choosing representative truth images')
            truth_dset = evaluator.sampler.dset

            # Choose representative images from each source dataset
            try:
                gid_to_source = {
                    gid: img.get('source', None)
                    for gid, img in truth_dset.imgs.items()
                }
                source_to_gids = ub.group_items(gid_to_source.keys(), gid_to_source.values())

                selected_gids = set()
                for source, _gids in source_to_gids.items():
                    selected = truth_dset.find_representative_images(_gids)
                    selected_gids.update(selected)

            except Exception:
                selected_gids = truth_dset.find_representative_images()

            dpath = ub.ensuredir((evaluator.paths['viz'], 'selected'))

            gid_to_true = coco_eval.gid_to_true
            for gid in ub.ProgIter(selected_gids, desc='draw selected imgs'):
                truth_dets = gid_to_true[gid]
                pred_dets = gid_to_pred[gid]

                thresh = 0.1
                if 'scores' in pred_dets.data:
                    pred_dets = pred_dets.compress(pred_dets.data['scores'] > thresh)
                # hack
                truth_dset.imgs[gid]['file_name'] = truth_dset.imgs[gid]['file_name'].replace('joncrall/data', 'joncrall/remote/namek/data')
                canvas = truth_dset.load_image(gid)
                canvas = truth_dets.draw_on(canvas, color='green')
                canvas = pred_dets.draw_on(canvas, color='blue')

                fig_fpath = join(dpath, 'eval-gid={}.jpg'.format(gid))
                kwimage.imwrite(fig_fpath, canvas)

        return results


if __name__ == '__main__':
    """
    Ignore:
        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=$HOME/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_vali.mscoco.json \
            --deployed=/home/joncrall/work/sealions/fit/runs/detect-sealion-cascade-v11/jwrqcohp/deploy_MM_CascadeRCNN_jwrqcohp_036_MHUOFO.zip

        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_test.mscoco.json \
            "--deployed=[\
                $HOME/remote/viame/work/bioharn/fit/nice/bioharn-det-mc-cascade-rgb-fine-coi-v40/torch_snapshots/_epoch_00000007.pt,\
                $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-fine-coi-v41/ufkqjjuk/torch_snapshots/_epoch_00000020.pt,\
            ]" \
            "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish,clapper" \
            --sampler_backend=cog --batch_size=16 --conf_thresh=0.1 --nms_thresh=0.8 --xpu=auto --window_overlap=0.5

        python ~/code/bioharn/bioharn/detect_eval.py \
            --dataset=$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_test.mscoco.json \
            "--deployed=[\
                $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgb-coi-v46/nngqryeh/deploy_MM_CascadeRCNN_nngqryeh_031_RVJZKO.zip,\
                $HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-fine-coi-v45/jsghbnij/deploy_MM_CascadeRCNN_jsghbnij_059_SXQKRF.zip,\
            ]" \
            "--classes_of_interest=live sea scallop,swimming sea scallop,flatfish,clapper" \
            --sampler_backend=cog --batch_size=16 --conf_thresh=0.1 --nms_thresh=0.8 --xpu=auto --window_overlap=0.5
    """
    evaluate_models()
