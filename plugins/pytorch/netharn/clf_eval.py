

def prototype_eval_clf():
    # hard coded prototype for classification evaluation
    import ubelt as ub
    from viame.pytorch.netharn import clf_predict

    config = {
        'dataset': ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_test.mscoco.json'),
        'sampler_workdir': ub.expandpath('~/work/bioharn'),
        'sampler_backend': 'auto',

        'deployed': ub.expandpath('$HOME/remote/namek/work/bioharn/fit/runs/bioharn-clf-rgb-v002/crloecin/deploy_ClfModel_crloecin_005_LSODSD.zip'),

        'workers': 4,
    }

    pred_config = ub.dict_isect(config, clf_predict.ClfPredictConfig.default)

    import kwcoco
    import ndsampler

    coco_dset = kwcoco.CocoDataset.coerce(config['dataset'])
    sampler = ndsampler.CocoSampler(
        coco_dset,
        workdir=config['sampler_workdir'],
        backend=config['sampler_backend'])

    predictor = clf_predict.ClfPredictor(pred_config)

    # TODO: probably need to make a cached predictor, prediction is pretty
    # quick though.
    # Cached prediction should maintain: aids, image ids, true categories if
    # known.
    result_gen = predictor.predict_sampler(sampler)
    results = list(result_gen)

    import numpy as np

    probs = np.array([result.data['prob'] for result in results])
    preds = np.array([result.data['cidx'] for result in results])
    confs = np.array([result.data['conf'] for result in results])

    classes = sampler.regions.classes
    idx_lut = sampler.regions.classes.id_to_idx
    true_cids = sampler.regions._targets['category_id']
    true_cidx = np.array([idx_lut[cid] for cid in true_cids])

    import kwarray
    data = kwarray.DataFrameArray({
        'pred': preds,
        'true': true_cidx,
        'score': confs,
    })

    from viame.pytorch.netharn.metrics import ConfusionVectors
    cfsn_vecs = ConfusionVectors(data, probs=probs, classes=classes)

    bin_vecs = cfsn_vecs.binarize_ovr()
    roc_result = bin_vecs.roc()['perclass']
    pr_result = bin_vecs.precision_recall()['perclass']

    roc_result.draw()
    coi = ['live sea scallop', 'dead sea scallop', 'clapper',
           'swimming sea scallop', 'roundfish', 'roundfish', 'flatfish']
    from kwcoco import metrics
    roc_coi = metrics.confusion_vectors.PerClass_ROC_Result(ub.dict_isect(roc_result.to_dict(), coi))
    pr_coi = metrics.confusion_vectors.PerClass_PR_Result(ub.dict_isect(pr_result.to_dict(), coi))

    import kwplot
    kwplot.figure(fnum=3, doclf=True)
    confusion = cfsn_vecs.confusion_matrix()
    kwplot.plot_matrix(confusion, fnum=3, showvals=0, logscale=True)

    roc_coi.draw(fnum=1)
    pr_coi.draw(fnum=2)

    import kwplot
    kwplot.autompl()

    report = cfsn_vecs.classification_report()
    print(ub.repr2(report['metrics']))
    print(report['metrics'].to_string())
