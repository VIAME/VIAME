import numpy as np
import json

@np.vectorize
def sigmoid(x):
    """Computes y = 1 / (1 + exp(-x)) in a numerically stable way.

    :param x: A real number.
    :return: The value 1 / (1 + exp(-x)).
    """
    if x < -30.0:
        return 0.0
    if x > 30.0:
        return 1.0
    return 1.0 / (1.0 + np.exp(-x))

def train_logistic_regression(features, labels, epochs=100, regularization=1e-3):
    import lasagne
    import theano
    import theano.tensor as T
    m, n = features.shape
    fx = theano.config.floatX
    X = T.matrix('X', fx)
    y = T.vector('y', fx)
    w = theano.shared(np.zeros((n,), fx), 'w')
    b = theano.shared(np.zeros((), fx), 'b')
    p = T.nnet.sigmoid(T.dot(X, w) + b)
    L = T.nnet.binary_crossentropy(p, y).mean()
    reg_L = L + 0.5 * regularization * T.sqr(w).sum()
    updates = lasagne.updates.adam(reg_L, [w, b])
    step = theano.function([], outputs=L, updates=updates, givens={X: features.astype(fx), y: labels.astype(fx)})
    loss = 0
    for _ in xrange(epochs):
        loss = step()
    parameter = np.hstack([w.get_value(), b.get_value()])
    return parameter, np.asscalar(loss)

def predict_logistic_regression(parameter, features):
    w = parameter[:-1]
    b = parameter[-1]
    return sigmoid(np.dot(features, w) + b)

if __name__ == '__main__':
    X = np.random.normal(size=(1000, 4))
    w0 = np.random.normal(size=(4,))
    b0 = np.random.normal()
    y = (np.dot(X, w0) + b0 >= 0)
    parameter, actual_loss = train_logistic_regression(X, y)
    print json.dumps(parameter.tolist(), indent = 4)
    print actual_loss
    print abs(np.round(predict_logistic_regression(parameter, X)) - y).sum() / len(X)

