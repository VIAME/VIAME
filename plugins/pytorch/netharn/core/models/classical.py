"""
WIP

This file should contain classes (that behave like torch models), but they
implement the learning of classical learning algorithms like SVM and
RandomForest.

Deep networks are amazing at learning features. However, I don't think it's
very useful to use linear logicstic regression as a classifier. In many cases I
think an SVM or a RandomForest might produce a superior classification model,
but this has yet to be shown.

TODO:
    - [ ] Classical Abstract API
    - [ ] Integration with the FitHarn
        - [ ] How do we swap netharn's backprop+SGD with sklearn's SVM and RandomForest fit methods?
        - [ ] Netharn needs a "classical" implementation of "run".
            - [ ] Simply use the data loader to load the data
            - [ ] Defaults should encourage use with deep features.
    - [ ] RandomForest
    - [ ] SVM
"""
import ubelt as ub


class ClassicalModule(ub.NiceRepr):
    """
    Abstraction for more classical approaches to learning classifiers
    """

    def __init__(self):
        raise NotImplementedError('abstract api')

    def fit(self, dataset):
        pass


class EstimatorFactory(ub.NiceRepr):
    """
    Returns sklearn classifier

    Example:
        >>> # xdoctest: +SKIP
        >>> from .models.classical import *  # NOQA
        >>> self = factory = EstimatorFactory('RF', verbose=1)
        >>> print('factory = {!r}'.format(factory))
        >>> estimator = factory()
        >>> print('estimator = {!r}'.format(estimator))
    """
    def __init__(self, clf_key='RF-OVR', **kw):
        self.clf_key = clf_key
        self.kw = kw
        self.est_kw1 = None
        self.est_kw2 = None
        self.est_kw = None
        self.est_type = None
        self.wrap_type = None
        self._normalize_attrs()

    def __nice__(self):

        kw_repr = ub.repr2(self.est_kw, nl=0)
        if len(kw_repr) > 50:
            kw_repr = ub.hash_data(kw_repr)[0:8]

        text = 'clf_key={}, est_kw={}'.format(
            self.clf_key, self.est_type, self.wrap_type, kw_repr
        )
        return text

    def __call__(self):
        clf = self._make_estimator()
        return clf

    def _normalize_attrs(self):
        tup = self.clf_key.split('-')
        wrap_type = None if len(tup) == 1 else tup[1]
        est_type = tup[0]

        self.est_type = est_type
        self.wrap_type = wrap_type

        est_kw1, est_kw2 = self._lookup_params(self.est_type)
        self.est_kw1 = est_kw1
        self.est_kw2 = est_kw2
        self.est_kw = ub.dict_union(est_kw1, est_kw2, self.kw)

    def _lookup_params(self, est_type):
        if est_type in {'RF', 'RandomForest'}:
            self.est_type = 'RF'
            est_kw1 = {
                # 'max_depth': 4,
                'bootstrap': True,
                'class_weight': None,
                'criterion': 'entropy',
                'max_features': 'sqrt',
                # 'max_features': None,
                'min_samples_leaf': 5,
                'min_samples_split': 2,
                # 'n_estimators': 64,
                'n_estimators': 256,
            }
            est_kw2 = {
                'random_state': 3915904814,
                'verbose': 0,
                'n_jobs': -1,
            }
        elif est_type in {'SVC', 'SVM'}:
            self.est_type = 'SVC'
            est_kw1 = dict(kernel='linear')
            est_kw2 = {}
        elif est_type in {'Logit', 'LogisticRegression'}:
            self.est_type = 'Logit'
            est_kw1 = {}
            est_kw2 = {}
        elif est_type in {'MLP'}:
            est_kw1 = dict(
                activation='relu', alpha=1e-05, batch_size='auto',
                beta_1=0.9, beta_2=0.999, early_stopping=False,
                epsilon=1e-08, hidden_layer_sizes=(10, 10),
                learning_rate='constant', learning_rate_init=0.001,
                max_iter=200, momentum=0.9, nesterovs_momentum=True,
                power_t=0.5, random_state=3915904814, shuffle=True,
                solver='lbfgs', tol=0.0001, validation_fraction=0.1,
                warm_start=False
            )
            est_kw2 = dict(verbose=False)
        else:
            raise KeyError('Unknown Estimator')
        return est_kw1, est_kw2

    def _make_estimator(self):
        make_estimator = self._make_est_func()
        clf = make_estimator()
        return clf

    def _make_est_func(self):
        import sklearn
        from sklearn import multiclass  # NOQA
        from sklearn import ensemble  # NOQA
        from sklearn import neural_network  # NOQA
        from sklearn import svm  # NOQA
        from sklearn import preprocessing  # NOQA
        from sklearn import pipeline  # NOQA
        from functools import partial

        wrap_type = self.wrap_type
        est_type = self.est_type

        multiclass_wrapper = {
            None: ub.identity,
            'OVR': sklearn.multiclass.OneVsRestClassifier,
            'OVO': sklearn.multiclass.OneVsOneClassifier,
        }[wrap_type]
        est_class = {
            'RF': sklearn.ensemble.RandomForestClassifier,
            'SVC': sklearn.svm.SVC,
            'Logit': partial(sklearn.linear_model.LogisticRegression, solver='lbfgs'),
            'MLP': sklearn.neural_network.MLPClassifier,
        }[est_type]

        est_kw = self.est_kw
        try:
            from sklearn.impute import SimpleImputer
            Imputer = SimpleImputer
            import numpy as np
            NAN = np.nan
        except Exception:
            from sklearn.preprocessing import Imputer
            NAN = 'NaN'
        if est_type == 'MLP':
            def make_estimator():
                pipe = sklearn.pipeline.Pipeline([
                    ('inputer', Imputer(
                        missing_values=NAN, strategy='mean')),
                    # ('scale', sklearn.preprocessing.StandardScaler),
                    ('est', est_class(**est_kw)),
                ])
                return multiclass_wrapper(pipe)
        elif est_type == 'Logit':
            def make_estimator():
                pipe = sklearn.pipeline.Pipeline([
                    ('inputer', Imputer(
                        missing_values=NAN, strategy='mean')),
                    ('est', est_class(**est_kw)),
                ])
                return multiclass_wrapper(pipe)
        else:
            def make_estimator():
                return multiclass_wrapper(est_class(**est_kw))

        return make_estimator


class ClfProblem(ClassicalModule):
    """
    Takes the place of model in Hyperparams

    CommandLine:
        xdoctest -m netharn.models.classical ClfProblem:1

    Example:
        >>> # xdoctest: +SKIP
        >>> from .models.classical import *  # NOQA
        >>> self = ClfProblem.demo('SVM-OVR', verbose=10)
        >>> print('self = {!r}'.format(self))
        >>> clf = self.fit()

    Example:
        >>> # xdoctest: +SKIP
        >>> from .models.classical import *  # NOQA
        >>> print(chr(10) * 3 + '------------')
        >>> clf = ClfProblem.demo('SVM-OVR', verbose=1).fit()
        >>> clf = ClfProblem.demo('SVM-OVO', verbose=1).fit()
        >>> print(chr(10) * 3 + '------------')
        >>> clf = ClfProblem.demo('RF-OVR', verbose=1).fit()
        >>> clf = ClfProblem.demo('RF-OVO', verbose=1).fit()
        >>> print(chr(10) * 3 + '------------')
        >>> clf = ClfProblem.demo('Logit-OVR', verbose=1).fit()
        >>> clf = ClfProblem.demo('Logit-OVO', verbose=1).fit()
        >>> print(chr(10) * 3 + '------------')
        >>> clf = ClfProblem.demo('MLP-OVR', verbose=1).fit()
        >>> clf = ClfProblem.demo('MLP-OVO', verbose=1).fit()
    """
    def __init__(self, clf_key='RF', data=None, **kw):
        self.factory = EstimatorFactory(clf_key, **kw)
        self.data = None

        if data is not None:
            self.connect_data(data)

    def __nice__(self):
        return '{}, data=({})'.format(
            self.factory.__nice__(),
            self.data.__nice__(),)

    def connect_data(self, data):
        self.data = data

    def fit(self):
        X = self.data.X
        y = self.data.y
        clf = self.factory()
        clf = clf.fit(X, y)
        return clf

    @classmethod
    def demo(cls, clf_key='RF', **kw):
        import sklearn.datasets
        import pandas as pd
        iris = sklearn.datasets.load_iris()

        X_df = pd.DataFrame(iris.data, columns=iris.feature_names)
        Y_df = pd.DataFrame({name: iris.target == idx
                             for idx, name in enumerate(iris.target_names)})

        classes = list(map(str, Y_df.columns))
        class_idxs = (Y_df.values * [[0, 1, 2]]).sum(axis=1)

        data = ClfData(X_df, class_idxs, classes)
        self = cls(clf_key, data=data, **kw)
        return self


class ClfData(ub.NiceRepr):
    def __init__(self, X, y, classes):
        self.X = X
        self.y = y
        self.classes = classes

    def __nice__(self):
        return '{}'.format(len(self.X))


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m netharn.models.classical all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
