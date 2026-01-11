"""
Modified from:
    https://github.com/ducha-aiki/LSUV-pytorch
"""
import numpy as np
import torch
import torch.nn.init
import torch.nn as nn
import ubelt as ub
from viame.pytorch.netharn import util
from viame.pytorch.netharn import api
from .functional import trainable_layers


def svd_orthonormal(shape, rng=None, cache_key=None):
    """
    If cache_key is specified, then the result will be cached, and subsequent
    calls with the same key and shape will return the same result.

    References:
        Orthonorm init code is taked from Lasagne
        https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    """
    rng = util.ensure_rng(rng)

    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))

    enabled = False and cache_key is not None
    if enabled:
        rand_sequence = rng.randint(0, 2 ** 16)
        depends = [shape, cache_key, rand_sequence]
        cfgstr = ub.hash_data(depends)
    else:
        cfgstr = ''

    # this process can be expensive, cache it

    # TODO: only cache very large matrices (4096x4096)
    # TODO: only cache very large matrices, not (256,256,3,3)
    cacher = ub.Cacher('svd_orthonormal', appname='netharn', enabled=enabled,
                       depends=cfgstr)
    q = cacher.tryload()
    if q is None:
        # print('Compute orthonormal matrix with shape ' + str(shape))
        a = rng.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        # print(shape, flat_shape)
        q = q.reshape(shape)
        q = q.astype(np.float32)
        cacher.save(q)
    return q


class Orthonormal(api.Initializer):
    def __init__(self, rng=None):
        self.rng = util.ensure_rng(rng)

    def forward(self, model):
        for name, m in trainable_layers(model, names=True):
            pass
            if isinstance(m, torch.nn.modules.conv._ConvNd) or isinstance(m, nn.Linear):
                if hasattr(m, 'weight_v'):
                    shape = tuple(m.weight_v.shape)
                    w_ortho = svd_orthonormal(shape, self.rng, cache_key=name)
                    m.weight_v.data[:] = torch.from_numpy(w_ortho)
                    try:
                        nn.init.constant_(m.bias, 0)
                    except Exception:
                        pass
                else:
                    shape = tuple(m.weight.shape)
                    w_ortho = svd_orthonormal(shape, self.rng, cache_key=name)
                    m.weight.data[:] = torch.from_numpy(w_ortho)
                    try:
                        nn.init.constant_(m.bias, 0)
                    except Exception:
                        pass
        return model


class LSUV(api.Initializer):
    """
    CommandLine:
        python -m netharn.initializers.lsuv LSUV:0

    Example:
        >>> # xdoc: +REQUIRES(--slow)
        >>> from .initializers.lsuv import *
        >>> import torchvision
        >>> import torch
        >>> #model = torchvision.models.AlexNet()
        >>> model = torchvision.models.SqueezeNet()
        >>> initer = LSUV(rng=0)
        >>> data = torch.autograd.Variable(torch.randn(4, 3, 224, 224))
        >>> initer.forward(model, data)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from .initializers.lsuv import *
        >>> import torchvision
        >>> import torch
        >>> model = torchvision.models.AlexNet()
        >>> initer = LSUV(rng=0)
        >>> data = torch.autograd.Variable(torch.randn(4, 3, 224, 224))
        >>> initer.forward(model, data)

    Example:
        >>> # DISABLE_DOCTEST
        >>> from .initializers.lsuv import *
        >>> import torchvision
        >>> import torch
        >>> model = torchvision.models.DenseNet()
        >>> initer = LSUV(rng=0)
        >>> data = torch.autograd.Variable(torch.randn(4, 3, 224, 224))
        >>> initer.forward(model, data)
    """
    def __init__(self, needed_std=1.0, std_tol=0.1, max_attempts=10,
                 do_orthonorm=True, rng=None):

        self.rng = util.ensure_rng(rng)

        self.do_orthonorm = do_orthonorm
        self.needed_std = needed_std
        self.std_tol = std_tol
        self.max_attempts = max_attempts

    def apply_weights_correction(self, m):
        if self.gg['hook'] is None:
            return
        if not self.gg['correction_needed']:
            return
        if (isinstance(m, torch.nn.modules.conv._ConvNd)) or (isinstance(m, nn.Linear)):
            if self.gg['counter_to_apply_correction'] < self.gg['hook_position']:
                self.gg['counter_to_apply_correction'] += 1
            else:
                if hasattr(m, 'weight_g'):
                    m.weight_g.data *= float(self.gg['current_coef'])
                    #print(m.weight_g.data)
                    #print(m.weight_v.data)
                    #print('weights norm after = ', m.weight.data.norm())
                    self.gg['correction_needed'] = False
                else:
                    #print('weights norm before = ', m.weight.data.norm())
                    m.weight.data *= self.gg['current_coef']
                    #print('weights norm after = ', m.weight.data.norm())
                    self.gg['correction_needed'] = False
                return
        return

    def store_activations(self, module, input, output):
        self.gg['act_dict'] = output.data.cpu().numpy()
        return

    def add_current_hook(self, m):
        if self.gg['hook'] is not None:
            return
        if (isinstance(m, torch.nn.modules.conv._ConvNd)) or (isinstance(m, nn.Linear)):
            #print('trying to hook to', m, self.gg['hook_position'], self.gg['done_counter'])
            if self.gg['hook_position'] > self.gg['done_counter']:
                self.gg['hook'] = m.register_forward_hook(self.store_activations)
                #print(' hooking layer = ', self.gg['hook_position'], m)
            else:
                #print(m, 'already done, skipping')
                self.gg['hook_position'] += 1
        return

    def count_conv_fc_layers(self, m):
        if (isinstance(m, torch.nn.modules.conv._ConvNd)) or (isinstance(m, nn.Linear)):
            self.gg['total_fc_conv_layers'] += 1
        return

    def remove_hooks(self, hooks):
        for h in hooks:
            h.remove()
        return

    def forward(self, model, data):
        import tqdm
        self.gg = {}
        self.gg['hook_position'] = 0
        self.gg['total_fc_conv_layers'] = 0
        self.gg['done_counter'] = -1
        self.gg['hook'] = None
        self.gg['act_dict'] = {}
        self.gg['counter_to_apply_correction'] = 0
        self.gg['correction_needed'] = False
        self.gg['current_coef'] = 1.0

        model.train(False)

        print('Starting LSUV')
        model.apply(self.count_conv_fc_layers)

        print('Total layers to process:', self.gg['total_fc_conv_layers'])
        if self.do_orthonorm:
            print('Applying orthogonal weights')
            Orthonormal(rng=self.rng).forward(model)
            # model.apply(self.orthogonal_weights_init)
            print('Orthonorm done')
            # if cuda:
            #     model = model.cuda()

        for layer_idx in tqdm.trange(self.gg['total_fc_conv_layers'], desc='init layer', leave=True):
            # print(layer_idx)
            model.apply(self.add_current_hook)
            out = model(data)  # NOQA
            current_std = self.gg['act_dict'].std()
            # tqdm.tqdm.write('layer {}: std={:.4f}'.format(layer_idx, current_std))
            #print  self.gg['act_dict'].shape
            attempts = 0
            for attempts in range(self.max_attempts):
                if not (np.abs(current_std - self.needed_std) > self.std_tol):
                    break
                self.gg['current_coef'] =  self.needed_std / (current_std  + 1e-8)
                self.gg['correction_needed'] = True
                model.apply(self.apply_weights_correction)

                # if cuda:
                #     model = model.cuda()

                out = model(data)  # NOQA

                current_std = self.gg['act_dict'].std()
                # tqdm.tqdm.write('layer {}: std={:.4f}, mean={:.4f}'.format(
                #         layer_idx, current_std, self.gg['act_dict'].mean()))
            if attempts + 1 >= self.max_attempts:
                tqdm.tqdm.write('Cannot converge in {} iterations'.format(self.max_attempts))
            if self.gg['hook'] is not None:
                self.gg['hook'].remove()
            self.gg['done_counter'] += 1
            self.gg['counter_to_apply_correction'] = 0
            self.gg['hook_position'] = 0
            self.gg['hook']  = None
            # print('finish at layer', layer_idx)
        print('LSUV init done!')

        # if not cuda:
        #     model = model.cpu()
        return model

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.initializers.lsuv
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
