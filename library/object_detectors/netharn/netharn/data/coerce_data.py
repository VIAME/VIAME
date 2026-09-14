"""
Copied from ndsampler
"""

import ubelt as ub


def coerce_datasets(config, build_hashid=False, verbose=1):
    """
    Coerce train / val / test datasets from standard netharn config keys

    TODO:
        * Does this belong in netharn?

    This only looks at the following keys in config:
        * datasets
        * train_dataset
        * vali_dataset
        * test_dataset

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> from .data.coerce_data import coerce_datasets
        >>> import kwcoco
        >>> config = {'datasets': 'special:shapes'}
        >>> print('config = {!r}'.format(config))
        >>> dsets = coerce_datasets(config)
        >>> print('dsets = {!r}'.format(dsets))

        >>> config = {'datasets': 'special:shapes256'}
        >>> coerce_datasets(config)

        >>> config = {
        >>>     'datasets': kwcoco.CocoDataset.demo('shapes'),
        >>> }
        >>> coerce_datasets(config)
        >>> coerce_datasets({
        >>>     'datasets': kwcoco.CocoDataset.demo('shapes'),
        >>>     'test_dataset': kwcoco.CocoDataset.demo('photos'),
        >>> })
        >>> coerce_datasets({
        >>>     'datasets': kwcoco.CocoDataset.demo('shapes'),
        >>>     'test_dataset': kwcoco.CocoDataset.demo('photos'),
        >>> })
    """
    # Ideally the user specifies a standard train/vali/test split
    def _rectify_fpath(key):
        fpath = key
        fpath = fpath.lstrip('path:').lstrip('PATH:')
        fpath = ub.expandpath(fpath)
        return fpath

    def _ensure_coco(coco):
        # Map a file path or an in-memory dataset to a CocoDataset
        import kwcoco
        from os.path import exists
        if coco is None:
            return None
        elif isinstance(coco, str):
            fpath = _rectify_fpath(coco)
            if exists(fpath):
                with ub.Timer('read kwcoco dataset: fpath = {!r}'.format(fpath)):
                    coco = kwcoco.CocoDataset(fpath, autobuild=False)
                print('building kwcoco index')
                coco._build_index()
            else:
                if not coco.lower().startswith('special:'):
                    import warnings
                    warnings.warn('warning start dataset codes with special:')
                    code = coco
                else:
                    code = coco.lower()[len('special:'):]
                coco = kwcoco.CocoDataset.demo(code)
        else:
            # print('live dataset')
            assert isinstance(coco, kwcoco.CocoDataset)
        return coco

    config = config.copy()

    subsets = {
        'train': config.get('train_dataset', None),
        'vali': config.get('vali_dataset', None),
        'test': config.get('test_dataset', None),
    }

    # specifying any train / vali / test disables datasets
    if any(d is not None for d in subsets.values()):
        config['datasets'] = None

    if verbose:
        print('[netharn.data.coerce_data] Checking for explicit subsets')
    subsets = ub.map_vals(_ensure_coco, subsets)

    # However, sometimes they just specify a single dataset, and we need to
    # make a split for it.
    # print('config = {!r}'.format(config))
    base = _ensure_coco(config.get('datasets', None))
    print('[netharn.data.coerce_data] base = {!r}'.format(base))
    if base is not None:
        if verbose:
            print('Splitting base into train/vali')
        # TODO: the actual split may need to be cached.
        factor = config.get('split_factor', 3)
        split_gids = _split_train_vali_test(base, factor=factor)
        if config.get('no_test', False):
            split_gids['train'] += split_gids.pop('test')
        for tag in split_gids.keys():
            gids = split_gids[tag]
            subset = base.subset(sorted(gids), copy=True)
            subset.tag = base.tag + '-' + tag
            subsets[tag] = subset

    subsets = {k: v for k, v in subsets.items() if v is not None}
    if build_hashid:
        print('Building subset hashids')
        for tag, subset in subsets.items():
            print('Build index for {}'.format(subset.tag))
            subset._build_index()
            print('Build hashid for {}'.format(subset.tag))
            subset._build_hashid(hash_pixels=False, verbose=10)

    # if verbose:
    #     print(_catfreq_columns_str(subsets))
    return subsets


def _print_catfreq_columns(subsets):
    print('Category Split Frequency:')
    print(_catfreq_columns_str(subsets))


def _catfreq_columns_str(subsets):
    import pandas as pd
    split_freq = {}
    for tag, subset in subsets.items():
        freq = subset.category_annotation_frequency()
        split_freq[tag] = freq

    df_ = pd.DataFrame.from_dict(split_freq)
    df_['sum'] = df_.sum(axis=1)
    df_ = df_.sort_values('sum')

    with pd.option_context('display.max_rows', 1000):
        text = df_.to_string()
    return text


def _split_train_vali_test(coco_dset, factor=3):
    """
    Args:
        factor (int): number of pieces to divide images into

    Example:
        >>> # xdoctest: +REQUIRES(module:ndsampler)
        >>> from .data.coerce_data import _split_train_vali_test
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('shapes8')
        >>> split_gids = _split_train_vali_test(coco_dset)
        >>> print('split_gids = {}'.format(ub.urepr(split_gids, nl=1)))
    """
    import kwarray
    images = coco_dset.images()

    def _stratified_split(gids, cids, n_splits=2, rng=None):
        """ helper to split while trying to maintain class balance within images """
        rng = kwarray.ensure_rng(rng)
        from ndsampler.utils import util_sklearn
        selector = util_sklearn.StratifiedGroupKFold(
            n_splits=n_splits, random_state=rng, shuffle=True)

        # from sklearn import model_selection
        # selector = model_selection.StratifiedKFold(
        #     n_splits=n_splits, random_state=rng, shuffle=True)
        skf_list = list(selector.split(X=gids, y=cids, groups=gids))
        trainx, testx = skf_list[0]

        if 0:
            _train_gids = set(ub.take(gids, trainx))
            _test_gids = set(ub.take(gids, testx))
            print('_train_gids = {!r}'.format(_train_gids))
            print('_test_gids = {!r}'.format(_test_gids))
        return trainx, testx

    # Create flat table of image-ids and category-ids
    gids, cids = [], []
    for gid_, cids_ in zip(images, images.annots.cids):
        cids.extend(cids_)
        gids.extend([gid_] * len(cids_))

    # Split into learn/test then split learn into train/vali
    learnx, testx = _stratified_split(gids, cids, rng=2997217409,
                                      n_splits=factor)
    learn_gids = list(ub.take(gids, learnx))
    learn_cids = list(ub.take(cids, learnx))
    _trainx, _valix = _stratified_split(learn_gids, learn_cids, rng=140860164,
                                        n_splits=factor)
    trainx = learnx[_trainx]
    valix = learnx[_valix]

    split_gids = {
        'train': sorted(ub.unique(ub.take(gids, trainx))),
        'vali': sorted(ub.unique(ub.take(gids, valix))),
        'test': sorted(ub.unique(ub.take(gids, testx))),
    }

    if True:
        # Hack to favor training a good model over testing it properly The only
        # real fix to this is to add more data, otherwise its simply a systemic
        # issue.
        split_gids['vali'] = sorted(set(split_gids['vali']) - set(split_gids['train']))
        split_gids['test'] = sorted(set(split_gids['test']) - set(split_gids['train']))
        split_gids['test'] = sorted(set(split_gids['test']) - set(split_gids['vali']))

    if __debug__:
        import itertools as it
        for a, b in it.combinations(split_gids.values(), 2):
            if (set(a) & set(b)):
                print('split_gids = {!r}'.format(split_gids))
                assert False

    return split_gids
