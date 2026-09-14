# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""The AdaBoost behind `process_query_adaboost`, in scikit-learn.

`iqr_session_adaboost` was `cv::ml::Boost` until P7-T09 and is this now. The
C++ side owns the IQR session -- the working index, the nearest-neighbour
expansion, the automatic negatives -- and calls in here for the four things
that were the model: fit, score, save and load.

This is a **different algorithm** and not a port of one. OpenCV boosts CART
trees with DISCRETE, REAL, LOGIT or GENTLE AdaBoost; scikit-learn's
`AdaBoostClassifier` is SAMME, and since 1.6 that is the only one it has. The
numbers move and the saved model changes format. That was the user's call,
and `design/STATUS.md` records it.

What it also does is **fix the ranking**. The OpenCV path asked for
`RAW_OUTPUT` meaning to get the weighted sum over the weak classifiers, and
that flag alone returns the class label instead, so every descriptor scored
0 or 1 and the ordered results were not ordered at all. `decision_function`
returns a real margin.

Determinism matters here, because the session's model is recorded and
because two queries over the same adjudications should not disagree.
`AdaBoostClassifier` passes its `random_state` down to each stump, where
scikit-learn's splitter uses it to break ties between equally good splits,
so leaving it unset makes training non-reproducible on tied data. It is
pinned.
"""
import io
import pickle

import numpy as np

# Pinned rather than left to the clock: see the note above on tie-breaking.
RANDOM_STATE = 0


def _estimator(n_estimators, max_depth):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    # `max_depth` 1 -- a decision stump -- is both the shipped configuration
    # and scikit-learn's own default base estimator, but it is passed
    # explicitly because the config key exists and means this.
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=int(max_depth),
                                         random_state=RANDOM_STATE),
        n_estimators=int(n_estimators),
        random_state=RANDOM_STATE,
    )


def train(features, labels, n_estimators=100, max_depth=1):
    """Fit on a list of descriptor vectors and their 0/1 labels.

    Returns the fitted estimator, or None when there is nothing to fit --
    fewer than two samples, or every sample of one class. The caller treats
    None as "no model" and falls back to similarity, which is what the C++
    did when `cv::ml::Boost::train` returned false.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int32)

    if x.ndim != 2 or x.shape[0] < 2 or len(np.unique(y)) < 2:
        return None

    model = _estimator(n_estimators, max_depth)
    model.fit(x, y)
    return model


def decision(model, vectors):
    """The margin for each vector: positive means the positive class.

    `decision_function` on a two-class `AdaBoostClassifier` gives one signed
    number per sample, which is the weighted vote difference -- the quantity
    the OpenCV path was asking for and not getting.
    """
    if model is None:
        return [0.0] * len(vectors)

    x = np.asarray(vectors, dtype=np.float64)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    return [float(value) for value in model.decision_function(x)]


def dumps(model):
    """The model as bytes, for the session to hand between query iterations.

    Pickle, which is what scikit-learn supports. It is not a portable format
    and it is not stable across scikit-learn versions -- neither was the
    OpenCV XML across OpenCV versions, and the blob never leaves the session
    that made it in any shipped pipeline. `loads` returns None rather than
    raising when it cannot read one.
    """
    if model is None:
        return b""

    buffer = io.BytesIO()
    pickle.dump(model, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    return buffer.getvalue()


def loads(data):
    """A model from `dumps`, or None if the bytes are not one."""
    if not data:
        return None

    try:
        model = pickle.loads(bytes(data))
    except Exception:
        return None

    # A pickle can hold anything; only take it if it is what we save
    if not hasattr(model, "decision_function"):
        return None

    return model
