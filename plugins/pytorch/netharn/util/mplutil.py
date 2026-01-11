# -*- coding: utf-8 -*-
"""
Mostly deprecated in favor of kwplot
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools as it
import numpy as np
import six
import ubelt as ub
from os.path import join, dirname


def pandas_plot_matrix(df, rot=90, ax=None, grid=True, label=None,
                       zerodiag=False,
                       cmap='viridis', showvals=False, logscale=True):
    import matplotlib as mpl
    import copy
    from matplotlib import pyplot as plt
    import matplotlib.cm  # NOQA
    import kwplot
    if ax is None:
        fig = kwplot.figure(fnum=1, pnum=(1, 1, 1))
        fig.clear()
        ax = plt.gca()
    ax = plt.gca()
    values = df.values
    if zerodiag:
        values = values.copy()
        values = values - np.diag(np.diag(values))

    # aximg = ax.imshow(values, interpolation='none', cmap='viridis')
    if logscale:
        from matplotlib.colors import LogNorm
        vmin = df[df > 0].min().min()
        norm = LogNorm(vmin=vmin, vmax=values.max())
    else:
        norm = None

    cmap = copy.copy(mpl.cm.get_cmap(cmap))  # copy the default cmap
    cmap.set_bad((0, 0, 0))

    aximg = ax.matshow(values, interpolation='none', cmap=cmap, norm=norm)
    # aximg = ax.imshow(values, interpolation='none', cmap='viridis', norm=norm)

    # ax.imshow(values, interpolation='none', cmap='viridis')
    ax.grid(False)
    cax = plt.colorbar(aximg, ax=ax)
    if label is not None:
        cax.set_label(label)

    ax.set_xticks(list(range(len(df.index))))
    ax.set_xticklabels([lbl[0:100] for lbl in df.index])
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rot)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment('center')

    ax.set_yticks(list(range(len(df.columns))))
    ax.set_yticklabels([lbl[0:100] for lbl in df.columns])
    for lbl in ax.get_yticklabels():
        lbl.set_horizontalalignment('right')
    for lbl in ax.get_yticklabels():
        lbl.set_verticalalignment('center')

    # Grid lines around the pixels
    if grid:
        offset = -.5
        xlim = [-.5, len(df.columns)]
        ylim = [-.5, len(df.index)]
        segments = []
        for x in range(ylim[1]):
            xdata = [x + offset, x + offset]
            ydata = ylim
            segment = list(zip(xdata, ydata))
            segments.append(segment)
        for y in range(xlim[1]):
            xdata = xlim
            ydata = [y + offset, y + offset]
            segment = list(zip(xdata, ydata))
            segments.append(segment)
        bingrid = mpl.collections.LineCollection(segments, color='w', linewidths=1)
        ax.add_collection(bingrid)

    if showvals:
        x_basis = np.arange(len(df.columns))
        y_basis = np.arange(len(df.index))
        x, y = np.meshgrid(x_basis, y_basis)

        for c, r in zip(x.flatten(), y.flatten()):
            val = df.iloc[r, c]
            ax.text(c, r, val, va='center', ha='center', color='white')
    return ax


def axes_extent(axs, pad=0.0):
    """
    Get the full extent of a group of axes, including axes labels, tick labels,
    and titles.
    """
    import matplotlib as mpl
    def axes_parts(ax):
        yield ax
        for label in ax.get_xticklabels():
            if label.get_text():
                yield label
        for label in ax.get_yticklabels():
            if label.get_text():
                yield label
        xlabel = ax.get_xaxis().get_label()
        ylabel = ax.get_yaxis().get_label()
        for label in (xlabel, ylabel, ax.title):
            if label.get_text():
                yield label

    items = it.chain.from_iterable(axes_parts(ax) for ax in axs)
    extents = [item.get_window_extent() for item in items]
    #mpl.transforms.Affine2D().scale(1.1)
    extent = mpl.transforms.Bbox.union(extents)
    extent = extent.expanded(1.0 + pad, 1.0 + pad)
    return extent


def extract_axes_extents(fig, combine=False, pad=0.0):
    """
    Extracts the extent of each axes item in inches. The main purpose of this
    is to set `bbox_inches` in `fig.savefig`, such that only the important data
    is visualized.

    Args:
        fig (Figure): the figure
        combine (bool): if True returns the union of each extent
        pad (float): additional padding around each axes

    Returns:
        matplotlib.transforms.Bbox or list of matplotlib.transforms.Bbox
    """
    # Make sure we draw the axes first so we can
    # extract positions from the text objects
    import matplotlib as mpl
    fig.canvas.draw()

    # Group axes that belong together
    atomic_axes = []
    seen_ = set([])
    for ax in fig.axes:
        if ax not in seen_:
            atomic_axes.append([ax])
            seen_.add(ax)

    dpi_scale_trans_inv = fig.dpi_scale_trans.inverted()
    axes_bboxes_ = [axes_extent(axs, pad) for axs in atomic_axes]
    axes_extents_ = [extent.transformed(dpi_scale_trans_inv) for extent in axes_bboxes_]
    # axes_extents_ = axes_bboxes_
    if combine:
        # Grab include extents of figure text as well
        # FIXME: This might break on OSX
        # http://stackoverflow.com/questions/22667224/bbox-backend
        renderer = fig.canvas.get_renderer()
        for mpl_text in fig.texts:
            bbox = mpl_text.get_window_extent(renderer=renderer)
            extent_ = bbox.expanded(1.0 + pad, 1.0 + pad)
            extent = extent_.transformed(dpi_scale_trans_inv)
            # extent = extent_
            axes_extents_.append(extent)
        axes_extents = mpl.transforms.Bbox.union(axes_extents_)
        # if True:
        #     axes_extents.x0 = 0
        #     # axes_extents.y1 = 0
    else:
        axes_extents = axes_extents_
    return axes_extents


def adjust_subplots(left=None, right=None, bottom=None, top=None, wspace=None,
                    hspace=None, fig=None):
    """
    Kwargs:
        left (float): left side of the subplots of the figure
        right (float): right side of the subplots of the figure
        bottom (float): bottom of the subplots of the figure
        top (float): top of the subplots of the figure
        wspace (float): width reserved for blank space between subplots
        hspace (float): height reserved for blank space between subplots
    """
    from matplotlib import pyplot as plt
    kwargs = dict(left=left, right=right, bottom=bottom, top=top,
                  wspace=wspace, hspace=hspace)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if fig is None:
        fig = plt.gcf()
    subplotpars = fig.subplotpars
    adjust_dict = subplotpars.__dict__.copy()
    del adjust_dict['validate']
    adjust_dict.update(kwargs)
    fig.subplots_adjust(**adjust_dict)


def render_figure_to_image(fig, dpi=None, transparent=None, **savekw):
    """
    Saves a figure as an image in memory.

    Args:
        fig (matplotlib.figure.Figure): figure to save

        dpi (int or str, Optional):
            The resolution in dots per inch.  If *None* it will default to the
            value ``savefig.dpi`` in the matplotlibrc file.  If 'figure' it
            will set the dpi to be the value of the figure.

        transparent (bool):
            If *True*, the axes patches will all be transparent; the
            figure patch will also be transparent unless facecolor
            and/or edgecolor are specified via kwargs.

        **savekw: other keywords passed to `fig.savefig`. Valid keywords
            include: facecolor, edgecolor, orientation, papertype, format,
            pad_inches, frameon.

    Returns:
        np.ndarray: an image in BGR or BGRA format.

    Notes:
        Be sure to use `fig.set_size_inches` to an appropriate size before
        calling this function.
    """
    import io
    import cv2
    # import matplotlib as mpl
    # axes_extents = extract_axes_extents(fig)
    # extent = mpl.transforms.Bbox.union(axes_extents)
    extent = 'tight'  # mpl might do this correctly these days
    with io.BytesIO() as stream:
        # This call takes 23% - 15% of the time depending on settings
        fig.savefig(stream, bbox_inches=extent, dpi=dpi,
                    transparent=transparent, **savekw)
        # fig.savefig(stream, **savekw)
        stream.seek(0)
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
    im_bgra = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    return im_bgra


def savefig2(fig, fpath, **kwargs):
    """
    Does a tight layout and saves the figure with transparency

    DEPRICATE
    """
    import matplotlib as mpl
    if 'transparent' not in kwargs:
        kwargs['transparent'] = True
    if 'extent' not in kwargs:
        axes_extents = extract_axes_extents(fig)
        extent = mpl.transforms.Bbox.union(axes_extents)
        kwargs['extent'] = extent
    fig.savefig(fpath, **kwargs)


def copy_figure_to_clipboard(fig):
    """
    References:
        https://stackoverflow.com/questions/17676373/python-matplotlib-pyqt-copy-image-to-clipboard
    """
    print('Copying figure %d to the clipboard' % fig.number)
    import cv2
    import matplotlib as mpl
    app = mpl.backends.backend_qt5.qApp
    QtGui = mpl.backends.backend_qt5.QtGui
    im_bgra = render_figure_to_image(fig, transparent=True)
    im_rgba = cv2.cvtColor(im_bgra, cv2.COLOR_BGRA2RGBA)
    im = im_rgba
    QImage = QtGui.QImage
    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGBA8888)
    clipboard = app.clipboard()
    clipboard.setImage(qim)

    # size = fig.canvas.size()
    # width, height = size.width(), size.height()
    # qim = QtGui.QImage(fig.canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32)

    # QtWidgets = mpl.backends.backend_qt5.QtWidgets
    # pixmap = QtWidgets.QWidget.grab(fig.canvas)
    # clipboard.setPixmap(pixmap)


def _get_axis_xy_width_height(ax=None, xaug=0, yaug=0, waug=0, haug=0):
    """ gets geometry of a subplot """
    from matplotlib import pyplot as plt
    if ax is None:
        ax = plt.gca()
    autoAxis = ax.axis()
    xy     = (autoAxis[0] + xaug, autoAxis[2] + yaug)
    width  = (autoAxis[1] - autoAxis[0]) + waug
    height = (autoAxis[3] - autoAxis[2]) + haug
    return xy, width, height


_LEGEND_LOCATION = {
    'upper right':  1,
    'upper left':   2,
    'lower left':   3,
    'lower right':  4,
    'right':        5,
    'center left':  6,
    'center right': 7,
    'lower center': 8,
    'upper center': 9,
    'center':      10,
}


_BASE_FNUM = 9001


def _save_requested(fpath_, save_parts):
    raise NotImplementedError('havent done this yet')
    # dpi = ub.argval('--dpi', type_=int, default=200)
    from os.path import expanduser
    from matplotlib import pyplot as plt
    dpi = 200
    fpath_ = expanduser(fpath_)
    print('Figure save was requested')
    # arg_dict = ut.get_arg_dict(prefix_list=['--', '-'],
    #                            type_hints={'t': list, 'a': list})
    arg_dict = {}
    # HACK
    arg_dict = {
        key: (val[0] if len(val) == 1 else '[' + ']['.join(val) + ']')
        if isinstance(val, list) else val
        for key, val in arg_dict.items()
    }
    fpath_ = fpath_.format(**arg_dict)
    fpath_ = fpath_.replace(' ', '').replace('\'', '').replace('"', '')
    dpath = ub.argval('--dpath', type_=str, default=None)
    if dpath is None:
        gotdpath = False
        dpath = '.'
    else:
        gotdpath = True

    fpath = join(dpath, fpath_)
    if not gotdpath:
        dpath = dirname(fpath_)
    print('dpath = %r' % (dpath,))

    fig = plt.gcf()
    fig.dpi = dpi

    fpath_strict = ub.expandpath(fpath)
    CLIP_WHITE = ub.argflag('--clipwhite')
    from netharn import util

    if save_parts:
        # TODO: call save_parts instead, but we still need to do the
        # special grouping.

        # Group axes that belong together
        atomic_axes = []
        seen_ = set([])
        for ax in fig.axes:
            div = _get_plotdat(ax, _DF2_DIVIDER_KEY, None)
            if div is not None:
                df2_div_axes = _get_plotdat_dict(ax).get('df2_div_axes', [])
                seen_.add(ax)
                seen_.update(set(df2_div_axes))
                atomic_axes.append([ax] + df2_div_axes)
                # TODO: pad these a bit
            else:
                if ax not in seen_:
                    atomic_axes.append([ax])
                    seen_.add(ax)

        hack_axes_group_row = ub.argflag('--grouprows')
        if hack_axes_group_row:
            groupid_list = []
            for axs in atomic_axes:
                for ax in axs:
                    groupid = ax.colNum
                groupid_list.append(groupid)

            groups = ub.group_items(atomic_axes, groupid_list)
            new_groups = list(map(ub.flatten, groups.values()))
            atomic_axes = new_groups
            #[[(ax.rowNum, ax.colNum) for ax in axs] for axs in atomic_axes]
            # save all rows of each column

        subpath_list = save_parts(fig=fig, fpath=fpath_strict,
                                  grouped_axes=atomic_axes, dpi=dpi)
        absfpath_ = subpath_list[-1]

        if CLIP_WHITE:
            for subpath in subpath_list:
                # remove white borders
                util.clipwhite_ondisk(subpath, subpath)
    else:
        savekw = {}
        # savekw['transparent'] = fpath.endswith('.png') and not noalpha
        savekw['transparent'] = ub.argflag('--alpha')
        savekw['dpi'] = dpi
        savekw['edgecolor'] = 'none'
        savekw['bbox_inches'] = extract_axes_extents(fig, combine=True)  # replaces need for clipwhite
        absfpath_ = ub.expandpath(fpath)
        fig.savefig(absfpath_, **savekw)

        if CLIP_WHITE:
            # remove white borders
            fpath_in = fpath_out = absfpath_
            util.clipwhite_ondisk(fpath_in, fpath_out)

    if ub.argflag(('--diskshow', '--ds')):
        # show what we wrote
        ub.startfile(absfpath_)


def save_parts(fig, fpath, grouped_axes=None, dpi=None):
    """
    FIXME: this works in mpl 2.0.0, but not 2.0.2

    Args:
        fig (?):
        fpath (str):  file path string
        dpi (None): (default = None)

    Returns:
        list: subpaths

    CommandLine:
        python -m draw_func2 save_parts

    Ignore:
        >>> # DISABLE_DOCTEST
        >>> import kwplot
        >>> kwplot.autompl()
        >>> import matplotlib as mpl
        >>> import matplotlib.pyplot as plt
        >>> def testimg(fname):
        >>>     return plt.imread(mpl.cbook.get_sample_data(fname))
        >>> fnames = ['grace_hopper.png', 'ada.png'] * 4
        >>> fig = plt.figure(1)
        >>> for c, fname in enumerate(fnames, start=1):
        >>>     ax = fig.add_subplot(3, 4, c)
        >>>     ax.imshow(testimg(fname))
        >>>     ax.set_title(fname[0:3] + str(c))
        >>>     ax.set_xticks([])
        >>>     ax.set_yticks([])
        >>> ax = fig.add_subplot(3, 1, 3)
        >>> ax.plot(np.sin(np.linspace(0, np.pi * 2)))
        >>> ax.set_xlabel('xlabel')
        >>> ax.set_ylabel('ylabel')
        >>> ax.set_title('title')
        >>> fpath = 'test_save_parts.png'
        >>> adjust_subplots(fig=fig, wspace=.3, hspace=.3, top=.9)
        >>> subpaths = save_parts(fig, fpath, dpi=300)
        >>> fig.savefig(fpath)
        >>> ub.startfile(subpaths[0])
        >>> ub.startfile(fpath)
    """
    if dpi:
        # Need to set figure dpi before we draw
        fig.dpi = dpi
    # We need to draw the figure before calling get_window_extent
    # (or we can figure out how to set the renderer object)
    # if getattr(fig.canvas, 'renderer', None) is None:
    fig.canvas.draw()

    # Group axes that belong together
    if grouped_axes is None:
        grouped_axes = []
        for ax in fig.axes:
            grouped_axes.append([ax])

    subpaths = []
    _iter = enumerate(grouped_axes, start=0)
    _iter = ub.ProgIter(list(_iter), label='save subfig')
    for count, axs in _iter:
        subpath = ub.augpath(fpath, suffix=chr(count + 65))
        extent = axes_extent(axs).transformed(fig.dpi_scale_trans.inverted())
        savekw = {}
        savekw['transparent'] = ub.argflag('--alpha')
        if dpi is not None:
            savekw['dpi'] = dpi
        savekw['edgecolor'] = 'none'
        fig.savefig(subpath, bbox_inches=extent, **savekw)
        subpaths.append(subpath)
    return subpaths


_qtensured = False


def _current_ipython_session():
    """
    Returns a reference to the current IPython session, if one is running
    """
    try:
        __IPYTHON__
    except NameError:
        return None
    else:
        import IPython
        ipython = IPython.get_ipython()
        # if ipython is None we must have exited ipython at some point
        return ipython


def qtensure():
    """
    If you are in an IPython session, ensures that your backend is Qt.
    """
    global _qtensured
    if not _qtensured:
        ipython = _current_ipython_session()
        if ipython:
            import sys
            if 'PyQt4' in sys.modules:
                ipython.magic('pylab qt4 --no-import-all')
                _qtensured = True
            else:
                ipython.magic('pylab qt5 --no-import-all')
                _qtensured = True


def aggensure():
    """
    Ensures that you are in agg mode as long as IPython is not running

    This might help prevent errors in tmux like:
        qt.qpa.screen: QXcbConnection: Could not connect to display localhost:10.0
        Could not connect to any X display.
    """
    import matplotlib as mpl
    current_backend = mpl.get_backend()
    if current_backend != 'agg':
        ipython = _current_ipython_session()
        if not ipython:
            import kwplot
            kwplot.set_mpl_backend('agg')


def colorbar(scalars, colors, custom=False, lbl=None, ticklabels=None,
             float_format='%.2f', **kwargs):
    """
    adds a color bar next to the axes based on specific scalars

    Args:
        scalars (ndarray):
        colors (ndarray):
        custom (bool): use custom ticks

    Kwargs:
        See plt.colorbar

    Returns:
        cb : matplotlib colorbar object

    Ignore:
        >>> import kwplot
        >>> kwplot.autompl()
        >>> scalars = np.array([-1, -2, 1, 1, 2, 7, 10])
        >>> cmap_ = 'plasma'
        >>> logscale = False
        >>> custom = True
        >>> reverse_cmap = True
        >>> val2_customcolor  = {
        ...        -1: UNKNOWN_PURP,
        ...        -2: LIGHT_BLUE,
        ...    }
        >>> colors = scores_to_color(scalars, cmap_=cmap_, logscale=logscale, reverse_cmap=reverse_cmap, val2_customcolor=val2_customcolor)
        >>> colorbar(scalars, colors, custom=custom)
        >>> df2.present()
        >>> show_if_requested()

    Ignore:
        >>> # ENABLE_DOCTEST
        >>> scalars = np.linspace(0, 1, 100)
        >>> cmap_ = 'plasma'
        >>> logscale = False
        >>> custom = False
        >>> reverse_cmap = False
        >>> colors = scores_to_color(scalars, cmap_=cmap_, logscale=logscale,
        >>>                          reverse_cmap=reverse_cmap)
        >>> colors = [lighten_rgb(c, .3) for c in colors]
        >>> colorbar(scalars, colors, custom=custom)
        >>> df2.present()
        >>> show_if_requested()
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm  # NOQA
    assert len(scalars) == len(colors), 'scalars and colors must be corresponding'
    if len(scalars) == 0:
        return None
    # Parameters
    ax = plt.gca()
    divider = _ensure_divider(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    xy, width, height = _get_axis_xy_width_height(ax)
    #orientation = ['vertical', 'horizontal'][0]
    TICK_FONTSIZE = 8
    #
    # Create scalar mappable with cmap
    if custom:
        # FIXME: clean this code up and change the name custom
        # to be meaningful. It is more like: display unique colors
        unique_scalars, unique_idx = np.unique(scalars, return_index=True)
        unique_colors = np.array(colors)[unique_idx]
        #max_, min_ = unique_scalars.max(), unique_scalars.min()
        #extent_ = max_ - min_
        #bounds = np.linspace(min_, max_ + 1, extent_ + 2)
        listed_cmap = mpl.colors.ListedColormap(unique_colors)
        #norm = mpl.colors.BoundaryNorm(bounds, listed_cmap.N)
        #sm = mpl.cm.ScalarMappable(cmap=listed_cmap, norm=norm)
        sm = mpl.cm.ScalarMappable(cmap=listed_cmap)
        sm.set_array(np.linspace(0, 1, len(unique_scalars) + 1))
    else:
        sorted_scalars = sorted(scalars)
        listed_cmap = scores_to_cmap(scalars, colors)
        sm = plt.cm.ScalarMappable(cmap=listed_cmap)
        sm.set_array(sorted_scalars)
    # Use mapable object to create the colorbar
    #COLORBAR_SHRINK = .42  # 1
    #COLORBAR_PAD = .01  # 1
    #COLORBAR_ASPECT = np.abs(20 * height / (width))  # 1

    cb = plt.colorbar(sm, cax=cax, **kwargs)

    ## Add the colorbar to the correct label
    #axis = cb.ax.yaxis  # if orientation == 'horizontal' else cb.ax.yaxis
    #position = 'bottom' if orientation == 'horizontal' else 'right'
    #axis.set_ticks_position(position)

    # This line alone removes data
    # axis.set_ticks([0, .5, 1])
    if custom:
        ticks = np.linspace(0, 1, len(unique_scalars) + 1)
        if len(ticks) < 2:
            ticks += .5
        else:
            # SO HACKY
            ticks += (ticks[1] - ticks[0]) / 2

        if isinstance(unique_scalars, np.ndarray) and unique_scalars.dtype.kind == 'f':
            ticklabels = [float_format % scalar for scalar in unique_scalars]
        else:
            ticklabels = unique_scalars
        cb.set_ticks(ticks)  # tick locations
        cb.set_ticklabels(ticklabels)  # tick labels
    elif ticklabels is not None:
        ticks_ = cb.ax.get_yticks()
        mx = ticks_.max()
        mn = ticks_.min()
        ticks = np.linspace(mn, mx, len(ticklabels))
        cb.set_ticks(ticks)  # tick locations
        cb.set_ticklabels(ticklabels)
        #cb.ax.get_yticks()
        #cb.set_ticks(ticks)  # tick locations
        #cb.set_ticklabels(ticklabels)  # tick labels
    # _set_plotdat(cb.ax, 'viztype', 'colorbar-%s' % (lbl,))
    # _set_plotdat(cb.ax, 'sm', sm)
    # FIXME: Figure out how to make a maximum number of ticks
    # and to enforce them to be inside the data bounds
    cb.ax.tick_params(labelsize=TICK_FONTSIZE)
    # Sets current axis
    plt.sca(ax)
    if lbl is not None:
        cb.set_label(lbl)
    return cb


_DF2_DIVIDER_KEY = '_df2_divider'


def _get_plotdat(ax, key, default=None):
    """ returns internal property from a matplotlib axis """
    _plotdat = _get_plotdat_dict(ax)
    val = _plotdat.get(key, default)
    return val


def _set_plotdat(ax, key, val):
    """ sets internal property to a matplotlib axis """
    _plotdat = _get_plotdat_dict(ax)
    _plotdat[key] = val


def _del_plotdat(ax, key):
    """ sets internal property to a matplotlib axis """
    _plotdat = _get_plotdat_dict(ax)
    if key in _plotdat:
        del _plotdat[key]


def _get_plotdat_dict(ax):
    """ sets internal property to a matplotlib axis """
    if '_plotdat' not in ax.__dict__:
        ax.__dict__['_plotdat'] = {}
    plotdat_dict = ax.__dict__['_plotdat']
    return plotdat_dict


def _ensure_divider(ax):
    """ Returns previously constructed divider or creates one """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = _get_plotdat(ax, _DF2_DIVIDER_KEY, None)
    if divider is None:
        divider = make_axes_locatable(ax)
        _set_plotdat(ax, _DF2_DIVIDER_KEY, divider)
        orig_append_axes = divider.append_axes
        def df2_append_axes(divider, position, size, pad=None, add_to_figure=True, **kwargs):
            """ override divider add axes to register the divided axes """
            div_axes = _get_plotdat(ax, 'df2_div_axes', [])
            new_ax = orig_append_axes(position, size, pad=pad, add_to_figure=add_to_figure, **kwargs)
            div_axes.append(new_ax)
            _set_plotdat(ax, 'df2_div_axes', div_axes)
            return new_ax
        new_method = df2_append_axes.__get__(divider, divider.__class__)
        setattr(divider, 'append_axes', new_method)
        # ut.inject_func_as_method(divider, df2_append_axes, 'append_axes', allow_override=True)
    return divider


def scores_to_cmap(scores, colors=None, cmap_='hot'):
    import matplotlib as mpl
    if colors is None:
        colors = scores_to_color(scores, cmap_=cmap_)
    scores = np.array(scores)
    colors = np.array(colors)
    sortx = scores.argsort()
    sorted_colors = colors[sortx]
    # Make a listed colormap and mappable object
    listed_cmap = mpl.colors.ListedColormap(sorted_colors)
    return listed_cmap


def scores_to_color(score_list, cmap_='hot', logscale=False, reverse_cmap=False,
                    custom=False, val2_customcolor=None, score_range=None,
                    cmap_range=(.1, .9)):
    """
    Other good colormaps are 'spectral', 'gist_rainbow', 'gist_ncar', 'Set1',
    'Set2', 'Accent'
    # TODO: plasma

    Args:
        score_list (list):
        cmap_ (str): defaults to hot
        logscale (bool):
        cmap_range (tuple): restricts to only a portion of the cmap to avoid extremes

    Returns:
        <class '_ast.ListComp'>

    Ignore:
        >>> ut.exec_funckw(scores_to_color, globals())
        >>> score_list = np.array([-1, -2, 1, 1, 2, 10])
        >>> # score_list = np.array([0, .1, .11, .12, .13, .8])
        >>> # score_list = np.linspace(0, 1, 100)
        >>> cmap_ = 'plasma'
        >>> colors = scores_to_color(score_list, cmap_)
        >>> imgRGB = kwarray.atleast_nd(np.array(colors)[:, 0:3], 3, tofront=True)
        >>> imgRGB = imgRGB.astype(np.float32)
        >>> imgBGR = kwimage.convert_colorspace(imgRGB, 'BGR', 'RGB')
        >>> imshow(imgBGR)
        >>> show_if_requested()

    Ignore:
        >>> score_list = np.array([-1, -2, 1, 1, 2, 10])
        >>> cmap_ = 'hot'
        >>> logscale = False
        >>> reverse_cmap = True
        >>> custom = True
        >>> val2_customcolor  = {
        ...        -1: UNKNOWN_PURP,
        ...        -2: LIGHT_BLUE,
        ...    }
    """
    import matplotlib.pyplot as plt
    assert len(score_list.shape) == 1, 'score must be 1d'
    if len(score_list) == 0:
        return []

    def apply_logscale(scores):
        scores = np.array(scores)
        above_zero = scores >= 0
        scores_ = scores.copy()
        scores_[above_zero] = scores_[above_zero] + 1
        scores_[~above_zero] = scores_[~above_zero] - 1
        scores_ = np.log2(scores_)
        return scores_

    if logscale:
        # Hack
        score_list = apply_logscale(score_list)
        #if loglogscale
        #score_list = np.log2(np.log2(score_list + 2) + 1)
    #if isinstance(cmap_, six.string_types):
    cmap = plt.get_cmap(cmap_)
    #else:
    #    cmap = cmap_
    if reverse_cmap:
        cmap = reverse_colormap(cmap)
    #if custom:
    #    base_colormap = cmap
    #    data = score_list
    #    cmap = customize_colormap(score_list, base_colormap)
    if score_range is None:
        min_ = score_list.min()
        max_ = score_list.max()
    else:
        min_ = score_range[0]
        max_ = score_range[1]
        if logscale:
            min_, max_ = apply_logscale([min_, max_])
    if cmap_range is None:
        cmap_scale_min, cmap_scale_max = 0., 1.
    else:
        cmap_scale_min, cmap_scale_max = cmap_range
    extent_ = max_ - min_
    if extent_ == 0:
        colors = [cmap(.5) for fx in range(len(score_list))]
    else:
        if False and logscale:
            # hack
            def score2_01(score):
                return np.log2(
                    1 + cmap_scale_min + cmap_scale_max *
                    (float(score) - min_) / (extent_))
            score_list = np.array(score_list)
            #rank_multiplier = score_list.argsort() / len(score_list)
            #normscore = np.array(list(map(score2_01, score_list))) * rank_multiplier
            normscore = np.array(list(map(score2_01, score_list)))
            colors =  list(map(cmap, normscore))
        else:
            def score2_01(score):
                return cmap_scale_min + cmap_scale_max * (float(score) - min_) / (extent_)
        colors = [cmap(score2_01(score)) for score in score_list]
        if val2_customcolor is not None:
            colors = [
                np.array(val2_customcolor.get(score, color))
                for color, score in zip(colors, score_list)]
    return colors


def interpolated_colormap(colors, resolution=64, space='lch-ab'):
    """
    Interpolates between colors in `space` to create a smooth listed colormap

    Args:
        colors (list or dict): list of colors or color objects and
            where in the map they should appear.

        resolution (int): number of discrete items in the colormap

        space (str): colorspace to interpolate in, using a CIE-LAB space will
            result in a perceptually uniform interpolation. HSV also works
            well.

    References:
        http://stackoverflow.com/questions/12073306/customize-colorbar-in-matplotlib

    CommandLine:
        python -m netharn.util.mplutil interpolated_colormap

    Example:
        >>> # DISABLE_DOCTEST
        >>> import kwplot
        >>> colors = [
        >>>     (0.0, kwplot.Color('green')),
        >>>     (0.5, kwplot.Color('gray')),
        >>>     (1.0, kwplot.Color('red')),
        >>> ]
        >>> space = 'lab'
        >>> #resolution = 16 + 1
        >>> resolution = 256 + 1
        >>> cmap = interpolated_colormap(colors, resolution, space)
        >>> # xdoc: +REQUIRES(--show)
        >>> import pylab
        >>> from matplotlib import pyplot as plt
        >>> a = np.linspace(0, 1, resolution).reshape(1, -1)
        >>> pylab.imshow(a, aspect='auto', cmap=cmap, interpolation='nearest')  # , origin="lower")
        >>> plt.grid(False)
        >>> show_if_requested()
    """
    import colorsys
    import matplotlib as mpl

    colors_inputs = colors

    if isinstance(colors_inputs, dict):
        colors_inputs = [(f, c) for f, c in sorted(colors_inputs.items())]
    else:
        if len(colors_inputs[0]) != 2:
            fracs = np.linspace(0, 1, len(colors_inputs))
            colors_inputs = list(zip(fracs, colors_inputs))

    # print('colors_inputs = {!r}'.format(colors_inputs))
    import kwplot
    colors = [kwplot.Color(c) for f, c in colors_inputs]
    fracs = [f for f, c in colors_inputs]

    basis = np.linspace(0, 1, resolution)
    fracs = np.array(fracs)
    indices = np.searchsorted(fracs, basis)
    indices = np.maximum(indices, 1)
    cpool = []

    from colormath import color_conversions
    # FIXME: need to ensure monkeypatch for networkx 2.0 in colormath
    # color_conversions._conversion_manager = color_conversions.GraphConversionManager()
    from colormath import color_objects
    def new_convertor(target_obj):
        source_obj = color_objects.sRGBColor
        def to_target(src_tup):
            src_tup = src_tup[0:3]
            src_co = source_obj(*src_tup)
            target_co = color_conversions.convert_color(src_co, target_obj)
            target_tup = target_co.get_value_tuple()
            return target_tup

        def from_target(target_tup):
            target_co = target_obj(*target_tup)
            src_co = color_conversions.convert_color(target_co, source_obj)
            src_tup = src_co.get_value_tuple()
            return src_tup
        return to_target, from_target

    def from_hsv(rgb):
        return colorsys.rgb_to_hsv(*rgb[0:3])

    def to_hsv(hsv):
        return colorsys.hsv_to_rgb(*hsv[0:3].tolist())

    classnames = {
        # 'AdobeRGBColor',
        # 'BaseRGBColor',
        'cmk': 'CMYColor',
        'cmyk': 'CMYKColor',
        'hsl': 'HSLColor',
        'hsv': 'HSVColor',
        'ipt': 'IPTColor',
        'lch-ab': 'LCHabColor',
        'lch-uv': 'LCHuvColor',
        'lab': 'LabColor',
        'luv': 'LuvColor',
        # 'SpectralColor',
        'xyz':  'XYZColor',
        # 'sRGBColor',
        'xyy': 'xyYColor'
    }

    conversions = {k: new_convertor(getattr(color_objects, v))
                   for k, v in classnames.items()}

    from_rgb, to_rgb = conversions['hsv']
    from_rgb, to_rgb = conversions['xyz']
    from_rgb, to_rgb = conversions['lch-uv']
    from_rgb, to_rgb = conversions['lch-ab']
    from_rgb, to_rgb = conversions[space]
    # from_rgb, to_rgb = conversions['lch']
    # from_rgb, to_rgb = conversions['lab']
    # from_rgb, to_rgb = conversions['lch-uv']

    for idx2, b in zip(indices, basis):
        idx1 = idx2 - 1
        f1 = fracs[idx1]
        f2 = fracs[idx2]

        c1 = colors[idx1].as01('rgb')
        c2 = colors[idx2].as01('rgb')
        # from_rgb, to_rgb = conversions['lch']
        h1 = np.array(from_rgb(c1))
        h2 = np.array(from_rgb(c2))
        alpha = (b - f1) / (f2 - f1)
        new_h = h1 * (1 - alpha) + h2 * (alpha)
        new_c = np.clip(to_rgb(new_h), 0, 1)
        # print('new_c = %r' % (new_c,))
        cpool.append(new_c)

    cpool = np.array(cpool)
    cmap = mpl.colors.ListedColormap(cpool, 'indexed')
    return cmap


def reverse_colormap(cmap):
    """
    References:
        http://nbviewer.ipython.org/github/kwinkunks/notebooks/blob/master/Matteo_colourmaps.ipynb
    """
    import matplotlib as mpl
    if isinstance(cmap,  mpl.colors.ListedColormap):
        return mpl.colors.ListedColormap(cmap.colors[::-1])
    else:
        reverse = []
        k = []
        for key, channel in six.iteritems(cmap._segmentdata):
            data = []
            for t in channel:
                data.append((1 - t[0], t[1], t[2]))
            k.append(key)
            reverse.append(sorted(data))
        cmap_reversed = mpl.colors.LinearSegmentedColormap(
            cmap.name + '_reversed', dict(zip(k, reverse)))
        return cmap_reversed


def draw_border(ax, color, lw=2, offset=None, adjust=True):
    'draws rectangle border around a subplot'
    if adjust:
        xy, width, height = _get_axis_xy_width_height(ax, -.7, -.2, 1, .4)
    else:
        xy, width, height = _get_axis_xy_width_height(ax)
    if offset is not None:
        xoff, yoff = offset
        xy = [xoff, yoff]
        height = - height - yoff
        width = width - xoff
    import matplotlib as mpl
    rect = mpl.patches.Rectangle(xy, width, height, lw=lw)
    rect = ax.add_patch(rect)
    rect.set_clip_on(False)
    rect.set_fill(False)
    import kwplot
    rect.set_edgecolor(kwplot.Color(color).as01('rgb'))
    return rect


def colorbar_image(domain, cmap='plasma', dpi=96, shape=(200, 20), transparent=False):
    """
    Notes:
        shape is approximate



    Ignore:
        domain = np.linspace(-30, 200)
        cmap='plasma'
        dpi = 80
        dsize = (20, 200)

        util.imwrite('foo.png', util.colorbar_image(np.arange(0, 1)), shape=(400, 80))

        import plottool as pt
        pt.qtensure()

        import matplotlib as mpl
        mpl.style.use('ggplot')
        util.imwrite('foo.png', util.colorbar_image(np.linspace(0, 1, 100), dpi=200, shape=(1000, 40), transparent=1))
        ub.startfile('foo.png')
    """
    import kwplot
    plt = kwplot.autoplt()

    fig = plt.figure(dpi=dpi)

    w, h = shape[1] / dpi, shape[0] / dpi
    # w, h = 1, 10
    fig.set_size_inches(w, h)

    ax = fig.add_subplot(1, 1, 1)

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap))
    sm.set_array(domain)

    plt.colorbar(sm, cax=ax)

    cb_img = render_figure_to_image(fig, dpi=dpi, transparent=transparent)

    plt.close(fig)

    return cb_img


def make_legend_img(classname_to_rgb, dpi=96, shape=(200, 200), mode='line',
                    transparent=False):
    """
    Makes an image of a categorical legend

    CommandLine:
        python -m netharn.util.mplutil make_legend_img

    Example:
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> classname_to_rgb = {
        >>>     'blue': kwplot.Color('blue').as01(),
        >>>     'red': kwplot.Color('red').as01(),
        >>> }
        >>> img = make_legend_img(classname_to_rgb)
        >>> # xdoctest: +REQUIRES(--show)
        >>> kwplot.autompl()
        >>> kwplot.imshow(img)
        >>> kwplot.show_if_requested()
    """
    import kwplot
    plt = kwplot.autoplt()

    def append_phantom_legend_label(label, color, type_='line', alpha=1.0, ax=None):
        if ax is None:
            ax = plt.gca()
        _phantom_legend_list = getattr(ax, '_phantom_legend_list', None)
        if _phantom_legend_list is None:
            _phantom_legend_list = []
            setattr(ax, '_phantom_legend_list', _phantom_legend_list)
        if type_ == 'line':
            phantom_actor = plt.Line2D((0, 0), (1, 1), color=color, label=label,
                                       alpha=alpha)
        else:
            phantom_actor = plt.Circle((0, 0), 1, fc=color, label=label,
                                       alpha=alpha)
        _phantom_legend_list.append(phantom_actor)

    fig = plt.figure(dpi=dpi)

    w, h = shape[1] / dpi, shape[0] / dpi
    fig.set_size_inches(w, h)

    ax = fig.add_subplot(1, 1, 1)
    for label, color in classname_to_rgb.items():
        append_phantom_legend_label(label, color, type_=mode, ax=ax)

    _phantom_legend_list = getattr(ax, '_phantom_legend_list', None)
    if _phantom_legend_list is None:
        _phantom_legend_list = []
        setattr(ax, '_phantom_legend_list', _phantom_legend_list)
    ax.legend(handles=_phantom_legend_list)
    ax.grid(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.axis('off')
    legend_img = render_figure_to_image(fig, dpi=dpi, transparent=transparent)
    plt.close(fig)
    return legend_img

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.util.mplutil
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
