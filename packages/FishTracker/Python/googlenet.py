# BLVC Googlenet, model from the paper:
# "Going Deeper with Convolutions"
# Original source:
# https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
# License: unrestricted use

# Download pretrained weights from:
# https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/blvc_googlenet.pkl


def build_model(input_var):
    from lasagne.layers import InputLayer
    from lasagne.layers import DenseLayer
    from lasagne.layers import ConcatLayer
    from lasagne.layers import NonlinearityLayer
    from lasagne.layers import GlobalPoolLayer
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
    from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
    from lasagne.layers import MaxPool2DLayer as PoolLayer
    from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
    from lasagne.nonlinearities import softmax, linear

    def build_inception_module(name, input_layer, nfilters):
        # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
        net = dict()
        net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
        net['pool_proj'] = ConvLayer(net['pool'], nfilters[0], 1, flip_filters=False)
        net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)
        net['3x3_reduce'] = ConvLayer(input_layer, nfilters[2], 1, flip_filters=False)
        net['3x3'] = ConvLayer(net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)
        net['5x5_reduce'] = ConvLayer(input_layer, nfilters[4], 1, flip_filters=False)
        net['5x5'] = ConvLayer(net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)
        net['output'] = ConcatLayer([net['1x1'], net['3x3'], net['5x5'], net['pool_proj']])
        return {'{}/{}'.format(name, k): v for k, v in net.items()}
    net = dict()
    net['input'] = InputLayer((None, 3, None, None), input_var)
    net['conv1/7x7_s2'] = ConvLayer(net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
    net['pool1/3x3_s2'] = PoolLayer(net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(net['pool1/norm1'], 64, 1, flip_filters=False)
    net['conv2/3x3'] = ConvLayer(net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = PoolLayer(net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)
    net.update(build_inception_module('inception_3a', net['pool2/3x3_s2'], [32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b', net['inception_3a/output'], [64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = PoolLayer(net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)
    net.update(build_inception_module('inception_4a', net['pool3/3x3_s2'], [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b', net['inception_4a/output'], [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c', net['inception_4b/output'], [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d', net['inception_4c/output'], [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e', net['inception_4d/output'], [128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = PoolLayer(net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)
    net.update(build_inception_module('inception_5a', net['pool4/3x3_s2'], [128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b', net['inception_5a/output'], [128, 384, 192, 384, 48, 128]))
    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
    net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'], num_units=1000, nonlinearity=linear)
    net['prob'] = NonlinearityLayer(net['loss3/classifier'], nonlinearity=softmax)
    return net


def load_pickle(path, mode='rb'):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    with open(path, mode) as f:
        return pickle.load(f)

def make_mapper(path):
    import os
    import numpy as np
    import lasagne
    import theano
    import theano.tensor as T

    input_var = T.tensor4('input_var', dtype=theano.config.floatX)
    net = build_model(input_var)
    blvc_googlenet = load_pickle(os.path.join(path, 'model/bvlc_googlenet.pkl'))
    pixel_mean = np.array([104, 117, 123], dtype=theano.config.floatX)
    lasagne.layers.set_all_param_values(net['prob'], blvc_googlenet['param values'])
    output_var = lasagne.layers.get_output(net['pool5/7x7_s1'])
    f = theano.function([input_var], output_var)

    def mapper(images):
        x = (np.array([image.transpose(2, 0, 1) for image in images], theano.config.floatX)
             - pixel_mean[None, :, None, None])
        y = f(x)
        return y

    return mapper

