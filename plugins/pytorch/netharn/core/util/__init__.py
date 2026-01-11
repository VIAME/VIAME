# -*- coding: utf-8 -*-
# flake8: noqa
"""
mkinit ~/code/kwarray/kwarray/__init__.py --relative --nomods -w
mkinit ~/code/kwimage/kwimage/__init__.py --relative --nomods -w
mkinit ~/code/kwplot/kwplot/__init__.py --relative --nomods -w
mkinit netharn.util --relative --nomods
"""
from __future__ import absolute_import, division, print_function, unicode_literals


def __getattr__(key):
    """
    Provide a module level getattr to give better error messages on deprecated
    and removed attributes

    Our deprecation errors will only work in Python 3.7+
    """
    from netharn.util._deprecation_helpers import _lookup_deprecated_attribute
    return _lookup_deprecated_attribute(key)


__submodules__ = [
    'imutil',
    'mplutil',
    'profiler',
    'util_averages',
    'util_filesys',
    'util_fname',
    'util_idstr',
    'util_inspect',
    'util_io',
    'util_iter',
    'util_json',
    'util_misc',
    'util_resources',
    'util_slider',
    'util_subextreme',
    'util_tensorboard',
    'util_torch',
    'util_zip',
]


from .imutil import (adjust_gamma, ensure_grayscale, get_num_channels,
                     image_slices, load_image_paths, overlay_colorized,
                     wide_strides_1d,)
from .mplutil import (adjust_subplots, aggensure, axes_extent, colorbar,
                      colorbar_image, copy_figure_to_clipboard, draw_border,
                      extract_axes_extents, interpolated_colormap,
                      make_legend_img, pandas_plot_matrix, qtensure,
                      render_figure_to_image, reverse_colormap, save_parts,
                      savefig2, scores_to_cmap, scores_to_color,)
from .profiler import (IS_PROFILING, profile, profile_now,)
from .util_averages import (CumMovingAve, ExpMovingAve, InternalRunningStats,
                            MovingAve, RunningStats, WindowedMovingAve,
                            absdev,)
from .util_filesys import (get_file_info,)
from .util_fname import (align_paths, check_aligned, dumpsafe,
                         shortest_unique_prefixes, shortest_unique_suffixes,)
from .util_idstr import (compact_idstr, make_idstr, make_short_idstr,)
from .util_inspect import (default_kwargs,)
from .util_io import (read_arr, read_h5arr, write_arr, write_h5arr,)
from .util_iter import (roundrobin,)
from .util_json import (LossyJSONEncoder, NumpyEncoder,
                        ensure_json_serializable, read_json, walk_json,
                        write_json,)
from .util_misc import (FlatIndexer, SupressPrint, align, align_lines,
                        strip_ansi,)
from .util_resources import (ensure_ulimit, resource_usage,)
from .util_slider import (SlidingWindow, Stitcher,)
from .util_subextreme import (argsubmax, argsubmaxima,)
from .util_tensorboard import (read_tensorboard_scalars,)
from .util_torch import (BatchNormContext, DisableBatchNorm,
                         IgnoreLayerContext, ModuleMixin, freeze_params,
                         grad_context, number_of_parameters, one_hot_embedding,
                         one_hot_lookup, torch_ravel_multi_index,
                         trainable_layers,)
from .util_zip import (split_archive, zopen,)

__all__ = ['BatchNormContext', 'CumMovingAve', 'DisableBatchNorm',
           'ExpMovingAve', 'FlatIndexer', 'IS_PROFILING', 'IgnoreLayerContext',
           'InternalRunningStats', 'LossyJSONEncoder',
           'ModuleMixin', 'MovingAve', 'NumpyEncoder', 'RunningStats',
           'SlidingWindow', 'Stitcher', 'SupressPrint', 'WindowedMovingAve',
           'absdev', 'adjust_gamma', 'adjust_subplots', 'aggensure', 'align',
           'align_lines', 'align_paths', 'argsubmax', 'argsubmaxima',
           'axes_extent', 'check_aligned', 'colorbar', 'colorbar_image',
           'compact_idstr', 'copy_figure_to_clipboard', 'default_kwargs',
           'draw_border', 'dumpsafe', 'ensure_grayscale',
           'ensure_json_serializable', 'ensure_ulimit', 'extract_axes_extents',
           'freeze_params', 'get_file_info', 'get_num_channels',
           'grad_context', 'image_slices', 'interpolated_colormap',
           'load_image_paths', 'make_idstr', 'make_legend_img',
           'make_short_idstr', 'number_of_parameters', 'one_hot_embedding',
           'one_hot_lookup', 'overlay_colorized', 'pandas_plot_matrix',
           'profile', 'profile_now', 'qtensure', 'read_arr', 'read_h5arr',
           'read_json', 'read_tensorboard_scalars', 'render_figure_to_image',
           'resource_usage', 'reverse_colormap', 'roundrobin', 'save_parts',
           'savefig2', 'scores_to_cmap', 'scores_to_color',
           'shortest_unique_prefixes', 'shortest_unique_suffixes',
           'split_archive', 'strip_ansi', 'torch_ravel_multi_index',
           'trainable_layers', 'walk_json', 'wide_strides_1d', 'write_arr',
           'write_h5arr', 'write_json', 'zopen']
