

def explicit_output_shape_for(model, inputs):
    """
    Registers forward hooks on a model, does an explicit forward pass, to
    measure output shapes.

    References:
        https://discuss.pytorch.org/t/how-to-register-forward-hooks-for-each-module/43347

    Example:
        from viame.arrows.pytorch.netharn import core as nh
        model = nh.models.ToyNet2d()
        inputs = torch.rand(1, 1, 256, 256)

        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> from .models.new_models_v1 import *  # NOQA
        >>> channels = ChannelSpec.coerce('rgb')
        >>> input_stats = None
        >>> self = MM_HRNetV2_w18_MaskRCNN(classes=3, channels=channels, with_mask=False)
        >>> model = self
        >>> inputs = batch = model.demo_batch()
        >>> type(inputs['inputs']['rgb'])
        >>> outputs = model(inputs)
        >>> type(inputs['inputs']['rgb'])

        explicit_output_shape_for(model, inputs)

    """
    from viame.arrows.pytorch.netharn.core.analytic.output_shape_for import OutputShape, HiddenShapes
    import torch
    import ubelt as ub

    # Create graph containing which modules are nested
    import networkx as nx
    modgraph = nx.DiGraph()

    for name, module in model.named_modules():
        modgraph.add_node(name)

    for name in modgraph.nodes.keys():
        modgraph.nodes[name]['hidden'] = HiddenShapes()
        modgraph.nodes[name]['output'] = None
        modgraph.nodes[name]['label'] = '.'.join(name.split('.')[-5:])

    for node in modgraph.nodes:
        parent = '.'.join(node.split('.')[:-1])
        if parent != node:
            modgraph.add_edge(parent, node)
            print('parent, node = {!r}, {!r}'.format(parent, node))

    print(nx.forest_str(modgraph))

    class CustomForwardHook(ub.NiceRepr):
        def __init__(hook, name=None):
            hook.name = name

        def __nice__(hook):
            return str(hook.name)

        def __call__(hook, module, inputs, outputs):
            print('\n--- HOOK ---')
            print('hook.name = {!r}'.format(hook.name))

            parent_nodes = modgraph.pred[hook.name]
            if len(parent_nodes):
                assert len(parent_nodes) == 1
                parent_name = ub.peek(parent_nodes)
                parent_node = modgraph.nodes[parent_name]
                parent_hidden = parent_node['hidden']
            else:
                parent_hidden = None

            name = hook.name
            node = modgraph.nodes[name]
            hidden = node['hidden']

            if isinstance(outputs, torch.Tensor):
                hidden['output'] = outputs.shape
                shape = OutputShape.coerce(outputs.shape, hidden=hidden)
            else:
                print(type(outputs))
                from viame.arrows.pytorch.netharn.core.data.data_containers import nestshape
                if isinstance(outputs, dict):
                    shape = nestshape(list(outputs.values()))
                else:
                    shape = nestshape(outputs)
                if shape is None:
                    print('outputs = {!r}'.format(outputs))

            output_shape = OutputShape.coerce(shape, hidden=hidden)

            if parent_hidden is not None:
                parent_hidden[hook.name] = output_shape
                print('parent_name = {!r}'.format(parent_name))
                # print('parent_hidden = {!r}'.format(parent_hidden))

            print('output_shape = {!r}'.format(output_shape))
            node['output'] = output_shape

    hooks = {}
    for name, module in model.named_modules():
        module._forward_hooks.clear()
        hook = CustomForwardHook(name)
        hooks[name] = module.register_forward_hook(hook)

    print('DOOIT')
    outputs = model(inputs)

    output_shape = modgraph.nodes['']['output']

    for node in modgraph.nodes:
        print('{}, {}'.format(node, modgraph.nodes[node]['output']))
        modgraph.nodes[node]['label'] = '{!r} {!r}'.format('.'.join(name.split('.')[-5:]), modgraph.nodes[node]['output'])

    print(nx.forest_str(modgraph))
    # parent = '.'.join(node.split('.')[:-1])

    print('output_shape.hidden = {}'.format(ub.repr2(output_shape.hidden.shallow(4), nl=-1)))
    print('output_shape.hidden = {}'.format(ub.repr2(output_shape.hidden.shallow(2), nl=-1)))

    output_shape.hidden['detector'].shallow(100)
