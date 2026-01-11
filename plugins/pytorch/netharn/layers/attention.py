"""
References:
    https://arxiv.org/pdf/1809.02983.pdf - Dual Attention Network for Scene Segmentation

    https://raw.githubusercontent.com/heykeetae/Self-Attention-GAN/master/sagan_models.py
"""
import torch
from torch import nn


class SelfAttention(nn.Module):
    """
    Self Attention Layer

    References:
    """

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_channels

        self.query_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels // 8,
                                    kernel_size=1)

        self.key_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=in_channels // 8, kernel_size=1)

        self.value_conv = nn.Conv2d(in_channels=in_channels,
                                    out_channels=in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args :
            x (Tensor): input feature maps (B x C x W x H)

        Returns :
            out : self attention value + input feature
            attention: B x N x N (N is Width*Height)
        """
        B, C, W, H = x.shape
        N = W * H

        proj_query = self.query_conv(x).view(B, -1, N).permute(0, 2, 1)  # B x C x(N)

        proj_key = self.key_conv(x).view(B, -1, N)  # B x C x (*W*H)

        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # B x (N) x (N)

        proj_value = self.value_conv(x).view(B, -1, N)  # B x C x N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x
        return out, attention


class ChannelAttention(nn.Module):
    """
    Channel attention module

    The channel attention module selectively emphasizes interdependent channel
    maps by integrating associated features among all channel map.

    Uses the uncentered scatter matrix (i.e. M @ M.T) to compute a unnormalized
    correlation-like matrix between channels.

    I think M @ M.T is an "uncentered scatter matrix"

    https://stats.stackexchange.com/questions/164997/relationship-between-gram-and-covariance-matrices

    not sure if this is the right term

    References:
        https://arxiv.org/pdf/1809.02983.pdf - Dual Attention Network for Scene Segmentation
        https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py

    Notes:
         Different from the position attention module, we directly calculate
         the channel attention map from the original features.

         Noted that we do not employ convolution layers to embed features
         before computing relationshoips of two channels, since it can maintain
         relationship between different channel maps. In addition, different
         from recent works [Zhang CVPR 2018 Context encoding for semantic
         segmentation] which explores channel relationships by a global pooling
         or encoding layer, we exploit spatial information at all corresponding
         positions to model channel correlations

    Ignore:

        >>> # Simple example to demonstrate why a multiplicative parameter
        >>> # at zero might or might not deviate to decrease the loss
        >>> x = torch.randn(10)
        >>> x[0] = -1000
        >>> p = nn.Parameter(torch.zeros(1) + 1e-1)
        >>> optim = torch.optim.SGD([p], lr=1e-1)
        >>> for i in range(10):
        >>>     loss = (x * (p ** 2)).sum()
        >>>     loss.backward()
        >>>     print('loss = {!r}'.format(loss))
        >>>     print('p.data = {!r}'.format(p.data))
        >>>     print('p.grad = {!r}'.format(p.grad))
        >>>     optim.step()
        >>>     optim.zero_grad()

        >>> # at zero might or might not deviate to decrease the loss
        >>> x = torch.randn(2)
        >>> x[0] = -1000
        >>> p = nn.Parameter(torch.zeros(1))
        >>> optim = torch.optim.SGD([p], lr=1e-1)
        >>> for i in range(10):
        >>>     loss = (x * p.clamp(0, None)).sum()
        >>>     loss.backward()
        >>>     print('loss = {!r}'.format(loss))
        >>>     print('p.data = {!r}'.format(p.data))
        >>>     print('p.grad = {!r}'.format(p.grad))
        >>>     optim.step()
        >>>     optim.zero_grad()

    Ignore:
        >>> B, C, H, W = 1, 3, 5, 7
        >>> inputs = torch.rand(B, C, H, W)
        >>> inputs = torch.arange(B * C * H * W).view(B, C, H, W).float()
        >>> self = ChannelAttention(C)
        >>> optim = torch.optim.SGD(self.parameters(), lr=1e-8)
        >>> for i in range(10):
        >>>     out = self(inputs)
        >>>     loss = (out.sum() ** 2)
        >>>     print('self.gamma = {!r}'.format(self.gamma))
        >>>     print('loss = {!r}'.format(loss))
        >>>     loss.backward()
        >>>     optim.step()
        >>>     optim.zero_grad()
    """
    def __init__(self, in_channels, attend_elsewhere=True):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels

        # hack to rectify the definiton in the paper with the implementaiton
        self.attend_elsewhere = attend_elsewhere

        # scale parameter (beta from paper)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): input feature maps (B, C, H, W)

        Returns:
            out (Tensor): attention value + input feature
            attention: (B, C, C)

        Example:
            >>> B, C, H, W = 1, 3, 5, 7
            >>> inputs = torch.rand(B, C, H, W)
            >>> self = ChannelAttention(C)
        """
        B, C, H, W = inputs.shape

        # Flatten spatial dims
        proj_query = inputs.view(B, C, -1)  # A
        proj_key = inputs.view(B, C, -1).permute(0, 2, 1)  # A.T
        proj_value = inputs.view(B, C, -1)  # A

        energy = torch.bmm(proj_query, proj_key)  # A @ A.T

        if self.attend_elsewhere:
            # Why the subtraction here?
            diag = torch.max(energy, dim=1, keepdim=True)[0].expand_as(energy)
            energy_new = diag - energy
            attention = energy_new.softmax(dim=1)
        else:
            attention = energy.softmax(dim=1)

        out = torch.bmm(attention, proj_value)
        out = out.view(B, C, H, W)

        residual = self.gamma * out
        out = residual + inputs
        return out
