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
    from netharn.models import densenet
    from netharn.models import descriptor_network
    from netharn.models import dual_path_net
    from netharn.models import resnet
    from netharn.models import toynet
    from netharn.models import yolo2

    from netharn.models.densenet import (DenseNet,)
    from netharn.models.descriptor_network import (DescriptorNetwork,)
    from netharn.models.dual_path_net import (Bottleneck, DPN, DPN26, DPN92,)
    from netharn.models.resnet import (ResNet,)
    from netharn.models.toynet import (ToyNet1d, ToyNet2d,)

    __all__ = ['Bottleneck', 'DPN', 'DPN26', 'DPN92', 'DenseNet',
               'DescriptorNetwork', 'ResNet', 'ToyNet1d', 'ToyNet2d', 'densenet',
               'descriptor_network', 'dual_path_net', 'resnet', 'toynet', 'yolo2']
