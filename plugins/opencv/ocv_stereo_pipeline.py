# -*- coding: utf-8 -*-
"""
Module for programatic Sprokit pipeline definitions

Outline / Proof-of-concept
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from os.path import normpath, expanduser, join, exists
from collections import OrderedDict as odict

import os
import sys

__all__ = ['Pipeline']

# ------
# Utilities (from ubelt)


def platform_cache_dir():
    """
    Returns a directory which should be writable for any application
    This should be used for temporary deletable data.
    """
    if sys.platform.startswith('win32'):  # nocover
        dpath_ = '~/AppData/Local'
    elif sys.platform.startswith('linux'):  # nocover
        dpath_ = '~/.cache'
    elif sys.platform.startswith('darwin'):  # nocover
        dpath_  = '~/Library/Caches'
    else:  # nocover
        raise NotImplementedError('Unknown Platform  %r' % (sys.platform,))
    dpath = normpath(expanduser(dpath_))
    return dpath


def ensure_app_cache_dir(appname, *args):
    """
    Returns a writable directory for an application and ensures the directory
    exists.  This should be used for temporary deletable data.

    Args:
        appname (str): the name of the application
        *args: any other subdirectories may be specified

    Returns:
        str: dpath: writable cache directory for this application
    """
    dpath = join(platform_cache_dir(), appname, *args)
    try:
        os.makedirs(dpath)
    except OSError:
        pass
    return dpath


def codeblock(block_str):
    """
    Helper (maybe use ubelt)

    Convinience function for defining code strings. Esspecially useful for
    templated code.
    """
    import textwrap
    return textwrap.dedent(block_str).strip('\n')


class NiceRepr(object):
    """
    Defines `__str__` and `__repr__` in terms of `__nice__` function
    Classes that inherit `NiceRepr` must define `__nice__`
    """
    def __repr__(self):
        try:
            classname = self.__class__.__name__
            devnice = self.__nice__()
            return '<%s(%s) at %s>' % (classname, devnice, hex(id(self)))
        except AttributeError:
            return object.__repr__(self)
            #return super(NiceRepr, self).__repr__()

    def __str__(self):
        try:
            classname = self.__class__.__name__
            devnice = self.__nice__()
            return '<%s(%s)>' % (classname, devnice)
        except AttributeError:
            import warnings
            warnings.warn('Error in __nice__ for %r' % (self.__class__,),
                          category=RuntimeWarning)
            raise

# ------
# Abstract classes


class Port(NiceRepr):
    """ abstract port """
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

    def __nice__(self):
        return self.name

    def absname(self):
        return '{}.{}'.format(self.parent.parent.name, self.name)


class PortSet(object):
    """ abstract ordered defaultdict-like container """
    def __init__(self, parent=None):
        self.parent = parent
        self.ports = odict()

    def __iter__(self):
        for iport in self.ports.values():
            yield iport

    def __getitem__(self, key):
        if key not in self.ports:
            self.ports[key] = self.wraped_port_type(key, self)
        return self.ports[key]


# ----
# Classes for internal representation


class IPort(Port):
    def __init__(self, name, parent):
        super(IPort, self).__init__(name, parent)
        self.connections = []

    def connect(self, oport):
        self.connections.append(oport)


class InputPortSet(PortSet):
    wraped_port_type = IPort

    def connect(self, mapping=None, **kwargs):
        if mapping is not None:
            kwargs.update(mapping)
        for iport_name, oport in kwargs.items():
            iport = self[iport_name]
            iport.connect(oport)


class OPort(Port):
    pass


class OutputPortSet(PortSet):
    wraped_port_type = OPort

# ----
# User facing class and API


class Process(NiceRepr):
    """
    Represents and maintains the definition of a pipeline node and its incoming
    and outgoing connections.
    """
    def __init__(self, type, name=None, config=None):
        self.type = type
        self.name = name
        self.config = config
        self.iports = InputPortSet(self)
        self.oports = OutputPortSet(self)

    def __nice__(self):
        return '{}::{}'.format(self.name, self.type)

    def make_node_text(self):
        """
        Creates a text based definition of this node for a .pipe file
        """
        fmtstr = codeblock(
            '''
            process {name}
              :: {type}
            ''')
        parts = [fmtstr.format(name=self.name, type=self.type)]
        if self.config:
            for key, val in self.config.items():
                parts.append('  :{key} {val}'.format(key=key, val=val))
        text = '\n'.join(parts)
        return text

    def make_edge_text(self):
        """
        Creates a text based definition of all incoming conections to this node
        for a .pipe file
        """
        fmtstr = codeblock(
            '''
            connect from {oport_abs_name}
                    to   {iport_abs_name}
            ''')
        parts = []
        for iport in self.iports:
            for oport in iport.connections:
                if oport is not None:
                    part = fmtstr.format(
                        oport_abs_name=oport.absname(),
                        iport_abs_name=iport.absname(),
                    )
                    parts.append(part)
        text = '\n'.join(parts)
        return text


class Pipeline(object):
    """
    Defines a Sprokit pipeline

    Example:
        >>> from define_pipeline import *  # NOQA
        >>> pipe = Pipeline()
        >>> # Pipeline nodes
        >>> input_image = pipe.add_process(
        ...     name='input_image', type='frame_list_input', config={
        ...         'image_list_file': 'input_list.txt',
        ...         'frame_time': 0.03333333,
        ...         'image_reader:type': 'ocv',
        ...     })
        >>> detector = pipe.add_process(
        ...     name='detector', type='hello_world_detector', config={
        ...         'text': 'Hello World!! (from python)',
        ...     })
        >>> # Connections
        >>> detector.iports.connect({
        ...     'image': detector.oports['image'],
        ... })
        >>> # Global config
        >>> pipe.config['_pipeline:_edge']['capacity'] = 5
        >>> pipe.config['_scheduler']['type'] = 'pythread_per_process'
        >>> # write the pipeline file to disk
        >>> pipe.write('hello_world_python.pipe')
        >>> # Draw the pipeline using graphviz
        >>> pipe.draw_graph('hello_world_python.png')
        >>> # Directly run the pipeline
        >>> pipe.run()
    """
    def __init__(self):
        # Store defined processes in a dictionary
        self.procs = odict()

        # Global configuration
        # TODO: determine the best way to represent this
        self.config = {
            '_pipeline:_edge': {
                # 'capacity': None,
            },
            '_scheduler': {
                # 'type': 'pythread_per_process',
            }
        }

    def add_process(self, type, name=None, config=None):
        """
        Adds a new process node to the pipeline.
        """
        assert name is not None, 'must specify name for now'
        node = Process(type=type, name=name, config=config)
        self.procs[name] = node
        return node

    def __getitem__(self, key):
        return self.procs[key]

    def make_global_text(self):
        """

        Ignore:
            # TODO: determine the best way to represent global configs
            # Ways end results can look:

            config _pipeline:_edge
                   :capacity 10

            config _scheduler
               :type pythread_per_process
        """
        # Note sure this is exactly how global configs are given yet
        lines = []
        for key, val in self.config.items():
            if val:
                lines.append('config {key}'.format(key=key))
                for key2, val2 in val.items():
                    lines.append('    :{key2} {val2}'.format(
                        key2=key2, val2=val2))
        text = '\n'.join(lines)
        return text

    def make_pipeline_text(self):
        blocks = []

        blocks.append(codeblock(
            '''
            # ----------------------
            # nodes
            #
            '''))
        for proc in self.procs.values():
            node_text = proc.make_node_text()
            if node_text:
                blocks.append(node_text)

        blocks.append(codeblock(
            '''
            # ----------------------
            # connections
            #
            '''))
        for proc in self.procs.values():
            edge_text = proc.make_edge_text()
            if edge_text:
                blocks.append(edge_text)

        blocks.append(codeblock(
            '''
            # ----------------------
            # global pipeline config
            #
            '''))
        blocks.append(self.make_global_text())

        text = '\n\n'.join(blocks)
        return text

    def write(self, fpath):
        print('writing pipeline filepath = {!r}'.format(fpath))
        text = self.make_pipeline_text()
        with open(fpath, 'w') as file:
            file.write(text)

    def run(self, dry=False):
        """
        Executes this pipeline.

        Writes a temporary pipeline file to your sprokit cache directory and
        calls the kwiver runner .
        """
        cache_dir = ensure_app_cache_dir('sprokit', 'temp_pipelines')
        # TODO make a name based on a hash of the text to avoid race conditions
        # or just use a builtin tempfile
        pipe_fpath = join(cache_dir, 'temp_pipeline_file.pipe')
        self.write(pipe_fpath)
        run_pipe_file(pipe_fpath, dry=dry)

    def to_networkx(self):
        """
        Creates a networkx representation of the process graph.

        Useful for visualization / any network graph analysis
        """
        import networkx as nx
        G = nx.DiGraph()
        # G.graph.update(self.config)

        if nx.__version__.startswith('1'):
            node_dict = G.node
        else:
            node_dict = G.nodes

        def _defaultstyle(node, color, shape='ellipse'):
            node_dict[node]['fillcolor'] = color
            node_dict[node]['style'] = 'filled'
            node_dict[node]['shape'] = shape
            # node_dict[node]['color'] = color

        # Add all processes
        # Make inputs and outputs nodes to prevent needing a multigraph
        for proc in self.procs.values():
            G.add_node(proc.name)
            _defaultstyle(proc.name, 'turquoise', shape='box')

            for iport in proc.iports:
                iport_name = iport.absname()
                G.add_node(iport_name)
                G.add_edge(iport_name, proc.name)
                node_dict[iport_name]['label'] = iport.name
                _defaultstyle(iport_name, 'red')

            for oport in proc.oports:
                oport_name = oport.absname()
                G.add_node(oport_name)
                G.add_edge(proc.name, oport_name)
                node_dict[oport_name]['label'] = oport.name
                _defaultstyle(oport_name, 'green')

        # Add all connections
        for proc in self.procs.values():
            for iport in proc.iports:
                iport_name = iport.absname()
                for oport in iport.connections:
                    if oport is not None:
                        oport_name = oport.absname()
                        G.add_edge(oport_name, iport_name)
        return G

    def draw_graph(self, fpath):
        """
        Draws the process graph using graphviz

        PreReqs:
            sudo apt-get install graphviz libgraphviz-dev pkg-config
            pip install networkx pygraphviz

            # fishlen_pipeline.py
        """
        import networkx as nx
        G = self.to_networkx()
        A = nx.nx_agraph.to_agraph(G)

        for proc in self.procs.values():
            nbunch = [proc.name]
            nbunch += [iport.absname() for iport in proc.iports]
            nbunch += [oport.absname() for oport in proc.oports]
            A.add_subgraph(nbunch, name='cluster_' + proc.name)
        A.layout(prog='dot')
        A.draw(fpath)


def find_kwiver_runner():
    """
    Search for the sprokit kwiver runner executable
    """
    # First check if kwiver runner is specified as an environment variable
    runner_fpath = os.environ.get('SPROKIT_PIPELINE_RUNNER', None)
    if runner_fpath is not None:
        return runner_fpath

    # If not, then search for the binary in the current dir and the PATH
    fnames = ['kwiver']
    if sys.platform.startswith('win32'):
        fnames.insert(0, 'kwiver.exe')

    search_paths = ['.']
    search_paths = os.environ.get('PATH', '').split(os.pathsep)

    for fname in fnames:
        for dpath in search_paths:
            fpath = join(dpath, fname)
            if os.path.isfile(fpath):
                return fpath


def run_pipe_file(pipe_fpath, dry=False):
    """
    Executes kwiver runner with a specific pipe file.
    """
    import os
    runner_fpath = find_kwiver_runner()
    if runner_fpath is None:
        raise Exception('Cannot find kwiver runner . Is it in your PATH?')

    print('found runner exe = {!r}'.format(runner_fpath))

    if not exists(pipe_fpath):
        raise IOError('Pipeline file {} does not exist'.format(pipe_fpath))

    if not exists(runner_fpath):
        raise NotImplementedError('Cannot find kwiver runner')

    command = '{} runner {}'.format(runner_fpath, pipe_fpath)
    print('command = "{}"'.format(command))
    if not dry:
        os.system(command)
