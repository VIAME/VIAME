from torch import nn
import torch
import torch.nn.functional as F


@torch.jit.script
def mish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return input * torch.tanh(F.softplus(input))


def beta_mish(input, beta=1.5):
    """
    Applies the β mish function element-wise:
        .. math::
            \\beta mish(x) = x * tanh(ln((1 + e^{x})^{\\beta}))
    See additional documentation for :mod:`echoAI.Activation.Torch.beta_mish`.

    References:
        https://github.com/digantamisra98/Echo/blob/master/echoAI/Activation/Torch/functional.py
    """
    return input * torch.tanh(torch.log(torch.pow((1 + torch.exp(input)), beta)))


class Mish_Function(torch.autograd.Function):

    """
    Applies the mish function element-wise:
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Plot:
    .. figure::  _static/mish.png
        :align:   center
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        https://github.com/digantamisra98/Echo/blob/master/echoAI/Activation/Torch/mish.py

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.tanh(F.softplus(x))  # x * tanh(ln(1 + exp(x)))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

    # else:
    #     @torch.jit.script
    #     def mish(input):
    #         delta = torch.exp(-input)
    #         alpha = 1 + 2 * delta
    #         return input * alpha / (alpha + 2 * delta * delta)


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input

    References:
        https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py
        https://github.com/thomasbrandon/mish-cuda
        https://arxiv.org/pdf/1908.08681v2.pdf

    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)

    Example:
        >>> x = torch.linspace(-20, 20, 100, requires_grad=True)
        >>> self = Mish()
        >>> y = self(x)
        >>> y.sum().backward()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.multi_plot(xydata={'beta=1': (x.data, y.data)}, fnum=1, pnum=(1, 2, 1))
        >>> kwplot.multi_plot(xydata={'beta=1': (x.data, x.grad)}, fnum=1, pnum=(1, 2, 2))
        >>> kwplot.show_if_requested()
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return Mish_Function.apply(input)
        # return mish(input)
