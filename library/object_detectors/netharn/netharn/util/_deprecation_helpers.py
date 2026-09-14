
def _lookup_deprecated_attribute(key):
    import ubelt as ub
    # mapping from module name to the attributes that were moved there.
    refactored = {
        'kwarray': [
            'ArrayAPI', 'DataFrameArray', 'DataFrameLight', 'FlatIndexer',
            'LocLight', 'RunningStats', 'apply_grouping', 'arglexmax',
            'argmaxima', 'argminima', 'atleast_nd', 'boolmask', 'ensure_rng',
            'group_consecutive', 'group_consecutive_indices', 'group_indices',
            'group_items', 'isect_flags', 'iter_reduce_ufunc',
            'maxvalue_assignment', 'mincost_assignment', 'mindist_assignment',
            'one_hot_embedding', 'one_hot_lookup', 'random_combinations',
            'random_product', 'seed_global', 'setcover', 'shuffle',
            'standard_normal', 'standard_normal32', 'standard_normal64',
            'stats_dict', 'uniform', 'uniform32'
        ],

        'kwimage': [
            'BASE_COLORS', 'Boxes', 'CSS4_COLORS', 'Color', 'Coords',
            'Detections', 'Heatmap', 'Mask', 'MaskList', 'MultiPolygon',
            'Points', 'PointsList', 'Polygon', 'PolygonList', 'Segmentation',
            'SegmentationList', 'TABLEAU_COLORS',
            'TORCH_GRID_SAMPLE_HAS_ALIGN', 'XKCD_COLORS', 'add_homog',
            'atleast_3channels', 'available_nms_impls', 'convert_colorspace',
            'daq_spatial_nms', 'decode_run_length', 'draw_boxes_on_image',
            'draw_clf_on_image', 'draw_line_segments_on_image',
            'draw_text_on_image', 'draw_vector_field', 'encode_run_length',
            'ensure_alpha_channel', 'ensure_float01', 'ensure_uint255',
            'fourier_mask', 'gaussian_patch', 'grab_test_image',
            'grab_test_image_fpath', 'imread', 'imresize', 'imscale',
            'imwrite', 'load_image_shape', 'make_channels_comparable',
            'make_heatmask', 'make_orimask', 'make_vector_field',
            'non_max_supression', 'normalize', 'num_channels',
            'overlay_alpha_images', 'overlay_alpha_layers',
            'radial_fourier_mask', 'remove_homog', 'rle_translate',
            'smooth_prob', 'stack_images', 'stack_images_grid',
            'subpixel_accum', 'subpixel_align', 'subpixel_getvalue',
            'subpixel_maximum', 'subpixel_minimum', 'subpixel_set',
            'subpixel_setvalue', 'subpixel_slice', 'subpixel_translate',
            'warp_image', 'warp_points', 'warp_tensor',
        ],

        'kwplot': [
            'BackendContext', 'Color', 'PlotNums', 'autompl', 'autoplt',
            'distinct_colors', 'distinct_markers', 'draw_boxes',
            'draw_boxes_on_image', 'draw_clf_on_image', 'draw_line_segments',
            'draw_points', 'draw_text_on_image', 'ensure_fnum', 'figure',
            'imshow', 'legend', 'make_conv_images', 'make_heatmask',
            'make_legend_img', 'make_orimask', 'make_vector_field',
            'multi_plot', 'next_fnum', 'plot_convolutional_features',
            'plot_matrix', 'plot_surface3d', 'set_figtitle', 'set_mpl_backend',
            'show_if_requested',
        ],

        'ubelt': [
            'CacheStamp',
        ]

    }
    ERROR_ON_ACCESS = True
    for modname, attrs in refactored.items():
        if key in attrs:
            text = ub.paragraph(
                '''
                The attribute `netharn.util.{key}` is deprecated.
                It was refactored and moved to `{modname}.{key}`.
                ''').format(key=key, modname=modname)
            if ERROR_ON_ACCESS:
                raise AttributeError(text)
            else:
                module = ub.import_module_from_name(modname)
                import warnings
                warnings.warn(text)
                return getattr(module, key)

    if key in ['SlidingIndexDataset', 'SlidingSlices']:
        raise AttributeError(
            'Deprecated {}, but still available in '
            'netharn.util.util_slider_dep'.format(key))

    raise AttributeError(key)
