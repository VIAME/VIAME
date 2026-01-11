"""
mkinit netharn.models
"""
# flake8: noqa

__DYNAMIC__ = False
if __DYNAMIC__:
    from mkinit import dynamic_init
    exec(dynamic_init(__name__))
else:
    # <AUTOGEN_INIT>
    from . import densenet
    from . import descriptor_network
    from . import dual_path_net
    from . import resnet
    from . import toynet
    from . import yolo2

    from .densenet import (DenseNet,)
    from .descriptor_network import (DescriptorNetwork,)
    from .dual_path_net import (Bottleneck, DPN, DPN26, DPN92,)
    from .resnet import (ResNet,)
    from .toynet import (ToyNet1d, ToyNet2d,)

    __all__ = ['Bottleneck', 'DPN', 'DPN26', 'DPN92', 'DenseNet',
               'DescriptorNetwork', 'ResNet', 'ToyNet1d', 'ToyNet2d', 'densenet',
               'descriptor_network', 'dual_path_net', 'resnet', 'toynet', 'yolo2']
