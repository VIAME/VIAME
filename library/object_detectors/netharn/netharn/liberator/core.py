"""
Extracts relevant parts of the source code.


The basic outline of how it works is this:

    * Given a reference to one or more function / classes

    * For each of these functions / classes:

        * look at the standalone source code via inspect

        * determine which names are undefined

        * Add definitions from the module that defined them.

    * Given a set of modules that should not be referenced

        * Go through each definition and remove those references

        * Look at name errors, and determine what code needs to be added

        * Recursively add / expand code until there are no more name errors.


Note:
    If the source code changes while the run is executing then this may not
    work correctly.

TODO:

- [x] Maintain a parse tree instead of raw lines

- [x] Keep a mapping from "definition names" to the top-level nodes in the parse tree that define them.

- [X] For each extracted node in the parse tree keep track of

    - [X] where it came from

    - [ ] what modifications were made to it

- [ ] Handle expanding imports nested within functions

- [ ] Maintain docstring formatting after using the node transformer


Issues:

    - [ ] We currently (0.0.1) get a KeyError in the case where, a module is imported like `import mod.submod` and all usage is of the form `mod.submod.attr`, then
"""
import ast
import astunparse
import copy
import inspect
import io
import sys
import ubelt as ub
import warnings
from os.path import isdir
from os.path import join
from os.path import basename
from os.path import abspath
from collections import OrderedDict

__all__ = ['Liberator', 'Closer']


__todo__ = """

# FIXME:
# The following case has misformatted docstrings and needs to handle
# the duplicate function name issue.

import ubelt as ub
import liberator
lib = liberator.Liberator()

import ubelt
import xdoctest
lib.add_dynamic(ubelt.util_import.modpath_to_modname)
lib.add_dynamic(ubelt.util_import.modname_to_modpath)
lib.add_dynamic(xdoctest.static_analysis.package_modpaths)
lib.expand(['ubelt', 'xdoctest'])
text = lib.current_sourcecode()
print(text)


"""


try:
    import rich
    rich_print = rich.print
except ImportError:
    rich_print = print


class LocalLogger:
    """
    A non-global logger used for specific code paths or class instances.
    """
    def __init__(self, tag='', verbose=0):
        self.verbose = verbose
        self.logs = []
        self.tag = tag
        self.indent = ''
        if verbose >= 2:
            self.debug('init new logger')

    def warn(self, msg):
        line = '[WARN.{}] '.format(self.tag) + self.indent + msg
        self.logs.append(line)
        if self.verbose >= 0:
            rich_print(line)

    def error(self, msg):
        line = '[ERROR.{}] '.format(self.tag) + self.indent + msg
        self.logs.append(line)
        if self.verbose >= 0:
            rich_print(line)

    def info(self, msg):
        line = '[INFO.{}] '.format(self.tag) + self.indent + msg
        self.logs.append(line)
        if self.verbose >= 1:
            rich_print(line)

    def debug(self, msg):
        line = '[DEBUG.{}] '.format(self.tag) + self.indent + msg
        self.logs.append(line)
        if self.verbose >= 2:
            rich_print(line)

    def _print_logs(self):
        rich_print('\n'.join(self.logs))

    @classmethod
    def coerce(cls, item, tag='', verbose=0):
        """
        Create logger from another logger
        """
        if isinstance(item, int):
            verbose = item

        self = cls(tag=tag, verbose=verbose)

        if isinstance(item, cls):
            # Make a sublogger, TODO: be more eloquent
            self.logs = item.logs
            self.verbose = item.verbose

        return self


class Liberator(ub.NiceRepr):
    r"""
    Maintains the current state of the source code

    There are 3 major steps:

    (a) extract the code to that defines a function or class from a module,

    (b) go back to the module and extract extra code required to define any
        names that were undefined in the extracted code, and

    (c) replace import statements to specified "expand" modules with the actual code
        used to define the variables accessed via the imports.

    This results in a standalone file that has absolutely no dependency on the
    original module or the specified "expand" modules (the expand module is
    usually the module that is doing the training for a network. This means
    that you can deploy a model independant of the training framework).

    Note:
        This is not designed to work for cases where the code depends on logic
        executed in a global scope (e.g. dynamically registering properties) .
        I think its actually impossible to statically account for this case in
        general.

    Args:
        tag (str): logging tag
        logger (Callable): logging function
        verbose (int): verbosity, 0 is nothing, 1 is info, 2 is debug, etc..

    Example:
        >>> import ubelt as ub
        >>> from liberator.core import Liberator
        >>> lib = Liberator(logger=print)
        >>> lib.add_dynamic(ub.find_exe, eager=False)
        >>> lib.expand(['ubelt'])
        >>> print(lib.current_sourcecode())

        >>> lib = Liberator()
        >>> lib.add_dynamic(ub.find_exe, eager=True)
        >>> print(lib.current_sourcecode())

        >>> lib = Liberator(logger=3, tag='mytest')
        >>> lib.add_dynamic(ub.Cacher, eager=True)
        >>> visitor = ub.peek(lib.visitors.values())
        >>> print('visitor.definitions = {}'.format(ub.urepr(ub.map_keys(str, visitor.definitions), nl=1)))
        >>> print('visitor.nested_definitions = {}'.format(ub.urepr(ub.map_keys(str, visitor.nested_definitions), nl=1)))

        >>> lib._print_logs()
        >>> lib.expand(['ubelt'])

    Ignore:
        from ubelt import _win32_links
        lib = Liberator()
        lib.add_dynamic(_win32_links._win32_symlink, eager=True)
        print(lib.current_sourcecode())
        definitions = list(visitor.definitions.values())
        import_defs = [d for d in definitions if 'Import' in d.type]
        print('import_defs = {}'.format(ub.urepr(import_defs, nl=1)))

    Example:
        >>> # xdoctest: +REQUIRES(module:fastai)
        >>> from liberator.core import *
        >>> import fastai.vision
        >>> obj = fastai.vision.models.WideResNet
        >>> expand_names = ['fastai']
        >>> lib = Liberator()
        >>> lib.add_dynamic(obj)
        >>> lib.expand(expand_names)
        >>> #print(ub.urepr(lib.body_defs, si=1))
        >>> print(lib.current_sourcecode())

    Example:
        >>> # xdoctest: +REQUIRES(module:fastai)
        >>> from liberator.core import Liberator
        >>> from fastai.vision.models import unet
        >>> lib = Liberator()
        >>> lib.add_dynamic(unet.DynamicUnet)
        >>> lib.expand(['fastai'])
        >>> print(lib.current_sourcecode())

    Example:
        >>> # xdoctest: +REQUIRES(module:netharn)
        >>> from liberator.core import *
        >>> import netharn as nh
        >>> from netharn.models.yolo2 import yolo2
        >>> obj = yolo2.Yolo2
        >>> expand_names = ['netharn']
        >>> lib = Liberator()
        >>> lib.add_static(obj.__name__, sys.modules[obj.__module__].__file__)
        >>> lib.expand(expand_names)
        >>> #print(ub.urepr(lib.body_defs, si=1))
        >>> print(lib.current_sourcecode())

    Example:
        >>> # Test a file with a nested dependency
        >>> import ubelt as ub
        >>> from liberator.core import Liberator
        >>> lib = Liberator(logger=print, verbose=3)
        >>> lib.add_dynamic(ub.download, eager=True)
        >>> visitor = ub.peek(lib.visitors.values())
        >>> print('visitor.definitions = {}'.format(ub.urepr(ub.map_keys(str, visitor.definitions), nl=1)))
        >>> print('visitor.nested_definitions = {}'.format(ub.urepr(ub.map_keys(str, visitor.nested_definitions), nl=1)))
        >>> print(lib.current_sourcecode())
        >>> lib.expand(['ubelt'])
        >>> print('visitor.definitions = {}'.format(ub.urepr(ub.map_keys(str, visitor.definitions), nl=1)))
        >>> print('visitor.nested_definitions = {}'.format(ub.urepr(ub.map_keys(str, visitor.nested_definitions), nl=1)))
        >>> print(lib.current_sourcecode())
    """
    def __init__(lib, tag='root', logger=None, verbose=0,
                 expand_internal_imports=False, minify=False):
        lib.minify = minify
        lib.header_defs = ub.odict()
        lib.body_defs = ub.odict()
        lib.visitors = {}
        lib.logger = LocalLogger.coerce(logger, tag=tag, verbose=verbose)
        lib.expand_internal_imports = expand_internal_imports

        lib._lazy_visitors = []

    def error(lib, msg):
        lib.logger.error(msg)

    def info(lib, msg):
        lib.logger.info(msg)

    def debug(lib, msg):
        lib.logger.debug(msg)

    def warn(lib, msg):
        lib.logger.warn(msg)

    def _print_logs(lib):
        lib.logger._print_logs()

    def __nice__(self):
        return self.logger.tag

    def _add_definition(lib, d):
        lib.debug('+ _add_definition = {!r}'.format(d))
        d = copy.deepcopy(d)
        # Hack to set the minify flag on a definition
        d.minify = lib.minify

        # print('ADD DEFINITION d = {!r}'.format(d))
        if 'Import' in d.type:
            if d.absname in lib.header_defs:
                del lib.header_defs[d.absname]
            lib.header_defs[d.absname] = d
        else:
            if d.absname in lib.body_defs:
                del lib.body_defs[d.absname]
            lib.body_defs[d.absname] = d
        lib.debug('+ finish _add_definition')

    def current_sourcecode(self):
        header_lines = [d.code for d in self.header_defs.values()]
        body_lines = [d.code for d in self.body_defs.values()][::-1]
        current_sourcecode = '\n'.join(header_lines)
        current_sourcecode += '\n\n\n'
        current_sourcecode += '\n\n\n'.join(body_lines)
        # current_sourcecode += '\n'.join(body_lines)
        return current_sourcecode

    def _ensure_visitor(lib, modpath=None, module=None):
        """
        Return an existing visitor for a module or create one if it doesnt
        exist
        """
        if modpath is None and module is not None:
            modpath = module.__file__

        if modpath not in lib.visitors:
            visitor = DefinitionVisitor.parse(
                module=module, modpath=modpath, logger=lib.logger)
            lib.visitors[modpath] = visitor
        visitor = lib.visitors[modpath]
        return visitor

    def add_dynamic(lib, obj, eager=True):
        """
        Add the source to define a live python object

        Args:
            obj (object): a reference to a class or function

            eager (bool): experimental. Keep as True for now.

        Example:
            >>> from liberator import core
            >>> import liberator
            >>> obj = core.unparse
            >>> eager = True
            >>> lib = liberator.Liberator()
            >>> lib.add_dynamic(obj, eager=eager)
            >>> print(lib.current_sourcecode())
        """
        lib.info('\n\n')
        lib.info('====\n\n')
        lib.info('START: lib.add_dynamic(obj={!r})'.format(obj))
        name = obj.__name__
        modname = obj.__module__
        module = sys.modules[modname]

        visitor = lib._ensure_visitor(module=module)

        d = visitor.extract_definition(name)

        lib._add_definition(d)

        if eager:
            lib.close(visitor)
        else:
            # Experimental
            lib._lazy_visitors.append(visitor)

        lib.info('FINISH: lib.add_dynamic(obj={!r})'.format(obj))

    def add_static(lib, name, modpath):
        """
        Statically extract a definition from a module file

        Args:
            name (str): the name of the member of the module to define

            modpath (PathLike): The path to the module

        Example:
            >>> from liberator import core
            >>> import liberator
            >>> modpath = core.__file__
            >>> name = core.unparse.__name__
            >>> lib = liberator.Liberator()
            >>> lib.add_static(name, modpath)
            >>> print(lib.current_sourcecode())
        """
        # print('ADD_STATIC name = {} from {}'.format(name, modpath))
        lib.info('lib.add_static(name={!r}, modpath={!r})'.format(name, modpath))

        visitor = lib._ensure_visitor(modpath=modpath)
        d = visitor.extract_definition(name)

        lib._add_definition(d)
        lib.close(visitor)

    def _lazy_close(lib):
        # Experimental
        lib.close2(ub.oset(lib._lazy_visitors))
        lib._lazy_visitors = []

    def close2(lib, visitors):
        """
        Experimental

        Populate all undefined names using the context from a module
        """
        # Parse the parent module to find only the relevant global varaibles and
        # include those in the extracted source code.
        lib.debug('closing')

        # Loop until all undefined names are defined
        names = True
        while names:
            # Determine if there are any variables needed from the parent scope
            current_sourcecode = lib.current_sourcecode()
            # Make sure we process names in the same order for hashability
            prev_names = names
            names = sorted(undefined_names(current_sourcecode))
            lib.debug(' * undefined_names = {}'.format(names))
            if names == prev_names:
                for visitor in visitors:
                    lib.debug('visitor.definitions = {}'.format(ub.urepr(
                        ub.map_keys(str, visitor.definitions), si=1, nl=1)))
                if 0:
                    warnings.warn('We were unable do do anything about undefined names')
                    return
                else:
                    # current_sourcecode = lib.current_sourcecode()
                    lib.error('--- <ERROR[4]> ---')
                    lib.error('Unable to define names')
                    lib.error(' * names = {!r}'.format(names))
                    #lib.error('<<< CURRENT_SOURCE >>>\n{}\n<<<>>>'.format(ub.highlight_code(current_sourcecode)))
                    lib.error('--- </ERROR[4]> ---')
                    raise AssertionError('unable to define names: {}'.format(names))
            for name in names:
                try:
                    # Greedilly choose the visitor that has the name we are
                    # looking for.
                    for visitor in visitors:
                        if name in visitor.definitions:
                            break
                    try:
                        lib.debug(' * try visitor.extract_definition({})'.format(name))
                        d = visitor.extract_definition(name)
                    except KeyError as ex:
                        lib.warn(' * encountered issue: {!r}'.format(ex))
                        # There is a corner case where we have the definition,
                        # we just need to move it to the top.
                        flag = False
                        for d_ in lib.body_defs.values():
                            if name == d_.name:
                                lib.warn(' * corner case: move definition to top')
                                lib._add_definition(d_)
                                flag = True
                                break
                        if not flag:
                            raise
                    else:
                        lib.debug(' * add extracted def {}'.format(name))
                        lib._add_definition(d)
                        # type_, text = visitor.extract_definition(name)
                except Exception as ex:
                    lib.warn(' * unable to extracted def {} due to {!r}'.format(name, ex))
                    # current_sourcecode = lib.current_sourcecode()
                    lib.error('--- <ERROR[3]> ---')
                    lib.error('Error computing source code extract_definition')
                    lib.error(' * failed to close name = {!r}'.format(name))
                    # lib.error('<<< CURRENT_SOURCE >>>\n{}\n<<<>>>'.format(ub.highlight_code(current_sourcecode)))
                    lib.error('--- </ERROR[3]> ---')

    def close(lib, visitor):
        """
        Populate all undefined names using the context from a module
        """
        # Parse the parent module to find only the relevant global varaibles and
        # include those in the extracted source code.
        lib.info('START close - i.e. populating, crawling')

        # Loop until all undefined names are defined
        names = True
        while names:
            # Determine if there are any variables needed from the parent scope
            current_sourcecode = lib.current_sourcecode()
            # Make sure we process names in the same order for hashability
            prev_names = names
            names = sorted(undefined_names(current_sourcecode))
            lib.debug(' * undefined_names = {}'.format(names))
            if names == prev_names:
                lib.debug('visitor.definitions = {}'.format(ub.urepr(
                    ub.map_keys(str, visitor.definitions), si=1, nl=1)))
                if 0:
                    warnings.warn('We were unable do do anything about undefined names')
                    return
                else:
                    # current_sourcecode = lib.current_sourcecode()
                    lib.error('--- <ERROR[1]> ---')
                    lib.error('Unable to define names')
                    lib.error(' * names = {!r}'.format(names))
                    #lib.error('<<< CURRENT_SOURCE >>>\n{}\n<<<>>>'.format(ub.highlight_code(current_sourcecode)))
                    lib.error('--- </ERROR[1]> ---')
                    raise AssertionError('unable to define names: {}'.format(names))
            for name in names:
                try:
                    try:
                        # pass
                        lib.debug(' * try visitor.extract_definition({!r})'.format(name))
                        d = visitor.extract_definition(name)
                    except KeyError as ex:
                        lib.debug(' * encountered issue: {!r}'.format(ex))
                        # There is a corner case where we have the definition,
                        # we just need to move it to the top.
                        flag = False
                        for d_ in lib.body_defs.values():
                            if name == d_.name:
                                lib.debug(' * corner case: move definition to top')
                                lib._add_definition(d_)
                                flag = True
                                break

                        # There is another corner case where we only have a
                        # prefix of the definition. Note, we could be more
                        # clever and look at the attribute usage in the current
                        # sourcefile instead of blindly taking everything with
                        # the given prefix.
                        if visitor.definitions.has_subtrie(name):
                            flag = True
                            for k, d in visitor.definitions.items(name):
                                lib.debug(' * add extracted prefix def {} for {}'.format(k, name))
                                lib._add_definition(d)

                        if not flag:
                            raise
                    else:
                        lib.debug(' * add extracted def {}'.format(name))
                        lib._add_definition(d)
                    # type_, text = visitor.extract_definition(name)
                except Exception as ex:
                    lib.warn(' * unable to extracted def {} due to {!r}'.format(name, ex))
                    # current_sourcecode = lib.current_sourcecode()
                    lib.error('--- <ERROR[2]> ---')
                    lib.error('Error computing source code extract_definition')
                    lib.error(' * failed to close name = {!r}'.format(name))
                    # lib.error('<<< CURRENT_SOURCE >>>\n{}\n<<<>>>'.format(ub.highlight_code(current_sourcecode)))
                    lib.error('--- </ERROR[2]> ---')
        lib.info('FINISH close')

    def expand(lib, expand_names):
        """
        Remove all references to specific modules by directly copying in the
        referenced source code. If the code is referenced from a module, then
        the references will need to change as well.

        Args:
            expand_name (List[str]): list of module names. For each module
                we expand any reference to that module in the closed source
                code by directly copying the referenced code into that file.
                This doesn't work in all cases, but it usually does.
                Reasons why this wouldn't work include trying to expand
                import from C-extension modules and expanding modules with
                complicated global-level logic.

        TODO:
            - [ ] Add special unique (mangled) suffixes to all expanded names
                to avoid name conflicts.

        Example:
            >>> # Test a heavier duty class
            >>> # xdoctest: +REQUIRES(module:netharn)
            >>> from liberator.core import *
            >>> import netharn as nh
            >>> obj = nh.device.MountedModel
            >>> #obj = nh.layers.ConvNormNd
            >>> #obj = nh.data.CocoDataset
            >>> #expand_names = ['ubelt', 'progiter']
            >>> expand_names = ['netharn']
            >>> lib = Liberator()
            >>> lib.add_dynamic(obj)
            >>> lib.expand(expand_names)
            >>> #print('header_defs = ' + ub.urepr(lib.header_defs, si=1))
            >>> #print('body_defs = ' + ub.urepr(lib.body_defs, si=1))
            >>> print('SOURCE:')
            >>> text = lib.current_sourcecode()
            >>> print(text)

        Example:
            >>> # Test a file with a nested dependency
            >>> import sys
            >>> if sys.version_info[0:2] <= (3, 8):
            ...     import pytest
            ...     pytest.skip('Internal import removal does not work on 3.8')
            >>> import ubelt as ub
            >>> from liberator.core import Liberator
            >>> lib = Liberator(logger=print, verbose=3, expand_internal_imports=True)
            >>> lib.add_dynamic(ub.download, eager=True)
            >>> lib.expand(['ubelt'])
            >>> print(lib.current_sourcecode())
        """
        lib.debug('\n\n')
        lib.debug('=== START EXPAND ===')

        if isinstance(expand_names, str):
            expand_names = [expand_names]  # hack for easier api

        if lib.expand_internal_imports:
            lib.debug('Experimental: expanding internal imports')
            # Experimental: force expansion of internal imports by removing
            # them, and pretending they were global imports. This probably is
            # not the 100% correct way to do it, but should be fine for a POC.
            for name in expand_names:
                internal_import_remover = RemoveInternalImports(name)
                for d in lib.body_defs.values():
                    internal_import_remover.update_definition(d)

                if internal_import_remover.rewritten_nodes:
                    lib.debug(f'internal_import_remover.rewritten_nodes={internal_import_remover.rewritten_nodes}')
                    for node in  internal_import_remover.rewritten_nodes:
                        removed_names = {a.name for a in node.names}
                        removed_names |= {a.asname for a in node.names}
                        # Hack: pretend the nested imports were header imports
                        found = 0
                        for visitor in lib.visitors.values():
                            for _d in visitor.nested_definitions.values():
                                if _d.name in removed_names:
                                    found = 1
                                    lib.header_defs[_d.absname] = _d
                        if not found:
                            lib.warn('Unable to expand internal node')

        lib.debug(f"expand_names: {expand_names} ")
        expand_iteration = 0
        # Expand references to internal modules
        still_needs_expansion = True
        while still_needs_expansion:
            expand_iteration += 1

            # Associate all top-level modules with any possible expand_name
            # that might trigger them to be expanded.
            # NOTE: this does not account for nested imports.
            expandable_definitions = ub.ddict(list)
            for d in lib.header_defs.values():
                parts = d.native_modname.split('.')
                for i in range(1, len(parts) + 1):
                    root = '.'.join(parts[:i])
                    expandable_definitions[root].append(d)

            lib.debug(f'Expand Iteration: {expand_iteration:03d}')
            lib.debug('Found: expandable_definitions = {!r}'.format(
                list(expandable_definitions.keys())))

            still_needs_expansion = False
            # current_sourcecode = lib.current_sourcecode()
            # closed_visitor = DefinitionVisitor.parse(source=current_sourcecode)
            for root in expand_names:
                needs_expansion = expandable_definitions.get(root, [])

                lib.debug('root = {!r}'.format(root))
                lib.debug('needs_expansion = {}'.format(ub.urepr(needs_expansion, nl=1)))

                d: Definition
                for d in needs_expansion:
                    if d._expanded:
                        continue
                    still_needs_expansion = True
                    # if d.absname == d.native_modname:
                    if ub.modname_to_modpath(d.absname):
                        lib.info('TODO: NEED TO CLOSE module = {}'.format(d))
                        # import warnings
                        # warnings.warn('Closing module {} may not be implemented'.format(d))
                        # definition is a module, need to expand its attributes
                        lib.expand_module_attributes(d)
                        d._expanded = True
                    else:
                        lib.info('TODO: NEED TO CLOSE attribute varname = {}'.format(d))
                        # warnings.warn('Closing attribute {} may not be implemented'.format(d))
                        # definition is a non-module, directly copy in its code
                        # We can directly replace this import statement by
                        # copy-pasting the relevant code from the other module
                        # (ASSUMING THERE ARE NO NAME CONFLICTS)

                        assert d.type == 'ImportFrom'

                        try:
                            native_modpath = ub.modname_to_modpath(d.native_modname)
                            if native_modpath is None:
                                raise Exception('Cannot find the module path for modname={!r}. '
                                                'Are you missing an __init__.py?'.format(d.native_modname))

                            sub_lib = Liberator(lib.logger.tag + '.sub.' + d.name,
                                                logger=lib.logger,
                                                expand_internal_imports=lib.expand_internal_imports)

                            try:
                                sub_lib.add_static(d.name, native_modpath)
                            except Exception:
                                ALLOW_FALLBACK = 1
                                if ALLOW_FALLBACK:
                                    # IF we somehow cant add the name
                                    # statically (e.g. the native-modpath
                                    # exposes the requested name dynamically,
                                    # so static extraction is impossible) we
                                    # may be able to hack around it by running
                                    # the code, and dynamically adding the
                                    # name.
                                    ns = {}
                                    exec(d.code, ns, ns)
                                    dynamic_obj = ns[d.name]
                                    sub_lib.add_dynamic(dynamic_obj)
                                else:
                                    raise

                            print(f'native_modpath={native_modpath}')
                            # sub_visitor = sub_lib.visitors[d.native_modname]
                            sub_lib.expand(expand_names)

                            # sub_lib.close(sub_visitor)
                        except NotAPythonFile as ex:
                            warnings.warn('CANNOT EXPAND d = {!r}, REASON: {}'.format(d, repr(ex)))
                            d._expanded = True
                            raise
                            continue
                        except Exception as ex:
                            warnings.warn('CANNOT EXPAND d = {!r}, REASON: {}'.format(d, repr(ex)))
                            d._expanded = True
                            raise
                            continue
                        else:
                            # Hack: remove the imported definition and add the
                            # explicit definition
                            # TODO: FIXME: more robust modification and replacement
                            d._code = '# LIBERATED: ' + d.code
                            d._expanded = True

                            for d_ in sub_lib.header_defs.values():
                                lib._add_definition(d_)
                            for d_ in sub_lib.body_defs.values():
                                lib._add_definition(d_)

                            # print('sub_visitor = {!r}'.format(sub_visitor))
                            # lib.close(sub_visitor)
                            lib.debug('CLOSED attribute d = {}'.format(d))

        lib.debug("=== FINISH EXPAND ===\n\n")

    def expand_module_attributes(lib, d):
        """
        Args:
            d (Definition): the definition to expand
        """
        # current_sourcecode = lib.current_sourcecode()
        # closed_visitor = DefinitionVisitor.parse(source=current_sourcecode)
        assert 'Import' in d.type
        varname = d.name
        varmodpath = ub.modname_to_modpath(d.absname)
        modname = d.absname

        def _exhaust(varname, modname, modpath):
            lib.debug('REWRITE ACCESSOR varname={!r}, modname={}, modpath={}'.format(varname, modname, modpath))

            # Modify the current node definitions and recompute code
            # TODO: make more robust
            rewriter = RewriteModuleAccess(varname)
            for d_ in lib.body_defs.values():
                rewriter.update_definition(d_)

            lib.debug('rewriter.accessed_attrs = {!r}'.format(rewriter.accessed_attrs))

            # For each modified attribute, copy in the appropriate source.
            for subname in rewriter.accessed_attrs:
                submodname = modname + '.' + subname
                submodpath = ub.modname_to_modpath(submodname)
                if submodpath is not None:
                    # if the accessor is to another module, exhaust until
                    # we reach a non-module
                    lib.debug('EXAUSTING: {}, {}, {}'.format(subname, submodname, submodpath))
                    _exhaust(subname, submodname, submodpath)
                else:
                    # Otherwise we can directly add the referenced attribute
                    lib.debug('FINALIZE: {} from {}'.format(subname, modpath))
                    lib.add_static(subname, modpath)

        _exhaust(varname, modname, varmodpath)
        d._code = '# ' + d.code

    def _experimental_write_to(self, fpath):
        """
        Write the current sourcode to a file, but preserve the header docstr
        """
        # import parso
        fpath = ub.Path(fpath)
        text = self.current_sourcecode()
        # new_module = parso.parse(text)

        if fpath.exists():
            import parso
            old_module = parso.parse(fpath.read_text())
            docstr_node = old_module.children[0]
            prefix = docstr_node.get_code()
            # .get_code()
        else:
            prefix = ''

        final = prefix + '\n' + text + '\n'
        fpath.write_text(final)


class UnparserVariant(astunparse.Unparser):
    """
    wraps astunparse to fix 2/3 compatibility minor issues

    Notes:
        x = np.random.rand(3, 3)
        # In python3 this works, but it fails in python2
        x[(..., 2)]
        # However, this works in both
        x[(Ellipsis, 2)]
        # Interestingly, this also works, but is not how astunparse generates code
        x[..., 2]
    """
    def _Ellipsis(self, t):
        # be compatible with python2 if possible
        self.write("Ellipsis")

    def _Constant(self, node):
        """
        Args:
            node (ast.Constant):
                a constant node, if it is a triplequote string we will try to
                make it look nice.

        Example:
            >>> from liberator.core import *  # NOQA
            >>> from liberator.core import UnparserVariant
            >>> tq = '"' * 3
            >>> code = ub.codeblock(
            >>>     fr'''
            >>>     def foobar():
            >>>         {tq}
            >>>         A docstring
            >>>         {tq}
            >>>     ''')
            >>> import ast
            >>> tree = ast.parse(code)
            >>> node = tree.body[0].body[0].value
            >>> v = io.StringIO()
            >>> self = UnparserVariant(node, file=v)
            >>> print(v.getvalue())
        """
        # Better support for multiline strings
        if not isinstance(node.value, str) or getattr(node, '_no_unparse_hack', 0) == 1:
            return super()._Constant(node)
        if node.lineno != node.end_lineno:
            # heuristic for tripple quote strings
            nl = '\n'
            tsq = "'''"
            tdq = '"""'
            # indent = ' ' * node.col_offset
            candidates = [
                # '"""\n' + ub.indent(node.s + '\n"""', ' ' * node.col_offset),
                # 'r"""\n' + ub.indent(node.s + '\n"""', ' ' * node.col_offset),
                # "'''\n" + ub.indent(node.s + "\n'''", ' ' * node.col_offset),
                # "r'''\n" + ub.indent(node.s + "\n'''", ' ' * node.col_offset),
                # 'r' + tdq + node.s + tdq,
                # 'r' + tsq + node.s + tsq,

                tdq + node.s + tdq,
                'r' + tdq + node.s + tdq,
                tsq + node.s + nl + tsq,
                'r' + tsq + node.s + tsq,
            ]
            if 0:
                for cand in candidates:
                    print(cand)
            found = None
            for cand in candidates:
                try:
                    reparsed = ast.literal_eval(cand)
                    if node.s != reparsed:
                        raise ValueError('not equivalent')
                except ValueError:
                    pass
                else:
                    found = cand
                    break

            if found:
                self.write(found)
            else:
                self.write(repr(node.s))
        else:
            self.write(repr(node.s))


def unparse(tree):
    r"""
    wraps astunparse to fix 2/3 compatibility minor issues

    Args:
        tree (ast.AST): abstract syntax tree to unparse

    FIXME:
        [ ] This needs to format docstrings better

    Example:
        >>> from liberator.core import *  # NOQA
        >>> tq = '"' * 3
        >>> code = ub.codeblock(
        >>>     fr'''
        >>>     def foobar():
        >>>         {tq}
        >>>         A docstring
        >>>         {tq}
        >>>     ''')
        >>> import ast
        >>> tree = ast.parse(code)
        >>> print(unparse(tree))
    """
    v = io.StringIO()
    # astunparse.Unparser(tree, file=v)
    UnparserVariant(tree, file=v)
    return v.getvalue()


def source_closure(obj, expand_names=[]):
    """
    Pulls the minimum amount of code needed to define `obj`.  Uses a
    combination of dynamic and static introspection.

    Args:
        obj (type): the class whose definition will be exported.

        expand_names (List[str]):
            EXPERIMENTAL. List of modules that should be expanded into raw
            source code.

    Returns:
        str: closed_sourcecode: text defining a new python module.

    CommandLine:
        xdoctest -m liberator.core source_closure

    Example:
        >>> # xdoctest: +REQUIRES(module:torchvision)
        >>> import torchvision
        >>> from torchvision import models
        >>> got = {}

        >>> model_class = models.AlexNet
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> got['alexnet'] = ub.hash_data(text)

        >>> model_class = models.DenseNet
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> got['densenet'] = ub.hash_data(text)

        >>> model_class = models.resnet50
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> got['resnet50'] = ub.hash_data(text)

        >>> model_class = models.Inception3
        >>> text = source_closure(model_class)
        >>> assert not undefined_names(text)
        >>> got['inception3'] = ub.hash_data(text)

        >>> # The hashes will depend on torchvision itself
        >>> if torchvision.__version__ == '0.2.1':
        >>>     # Note: the hashes may change if the exporter changes formats
        >>>     want = {
        >>>         'alexnet': '4b2ab9c8e27b34602bdff99cbc',
        >>>         'densenet': 'fef4788586d2b93587ec52dd9',
        >>>         'resnet50': '343e6a73e754557fcce3fdb6',
        >>>         'inception3': '2e43a58133d0817753383',
        >>>     }
        >>>     failed = []
        >>>     for k in want:
        >>>         if not got[k].startswith(want[k]):
        >>>             item = (k, got[k], want[k])
        >>>             print('failed item = {!r}'.format(item))
        >>>             failed.append(item)
        >>>     assert not failed, str(failed)
        >>> else:
        >>>     warnings.warn('Unsupported version of torchvision')

    Example:
        >>> # Test a heavier duty class
        >>> # xdoctest: +REQUIRES(module:netharn)
        >>> from liberator.core import *
        >>> import netharn as nh
        >>> obj = nh.layers.ConvNormNd
        >>> expand_names = ['netharn']
        >>> text = source_closure(obj, expand_names)
        >>> print(text)
    """
    lib = Liberator()

    # First try to add statically (which tends to be slightly nicer)
    try:
        try:
            name = obj.__name__
            modpath = sys.modules[obj.__module__].__file__
        except Exception:
            # Otherwise add dynamically
            lib.add_dynamic(obj)
        else:
            lib.add_static(name, modpath)
        if expand_names:
            lib.expand(expand_names)
        closed_sourcecode = lib.current_sourcecode()
    except Exception:
        print('ERROR IN CLOSING')
        print('[[[ START CLOSE LOGS ]]]')
        print('lib.logs =\n{}'.format('\n'.join(lib.logger.logs)))
        print('[[[ END CLOSE LOGS ]]]')
        raise
    return closed_sourcecode


def _parse_static_node_value(node):
    """
    Extract a constant value from a node if possible
    """
    if isinstance(node, ast.Num):
        value = node.n
    elif isinstance(node, ast.Str):
        value = node.s
    elif isinstance(node, ast.List):
        value = list(map(_parse_static_node_value, node.elts))
    elif isinstance(node, ast.Tuple):
        value = tuple(map(_parse_static_node_value, node.elts))
    elif isinstance(node, (ast.Dict)):
        keys = map(_parse_static_node_value, node.keys)
        values = map(_parse_static_node_value, node.values)
        value = OrderedDict(zip(keys, values))
        # value = dict(zip(keys, values))
    elif isinstance(node, (ast.NameConstant)):
        value = node.value
    else:
        msg = ('Cannot parse a static value from non-static node '
               'of type: {!r}'.format(type(node)))
        # print('node.__dict__ = {!r}'.format(node.__dict__))
        # print('msg = {!r}'.format(msg))
        raise TypeError(msg)
    return value


def undefined_names(sourcecode):
    """
    Parses source code for undefined names

    Args:
        sourcecode (str): code to check for unused names

    Returns:
        Set[str]: the unused variable names

    Example:
        >>> # xdoctest: +REQUIRES(module:pyflakes)
        >>> print(ub.urepr(undefined_names('x = y'), nl=0))
        {'y'}
    """
    import pyflakes.api
    import pyflakes.reporter

    class CaptureReporter(pyflakes.reporter.Reporter):
        def __init__(reporter, warningStream, errorStream):
            reporter.syntax_errors = []
            reporter.messages = []
            reporter.unexpected = []

        def unexpectedError(reporter, filename, msg):
            reporter.unexpected.append(msg)

        def syntaxError(reporter, filename, msg, lineno, offset, text):
            reporter.syntax_errors.append(msg)

        def flake(reporter, message):
            reporter.messages.append(message)

    names = set()

    reporter = CaptureReporter(None, None)
    pyflakes.api.check(sourcecode, '_.py', reporter)
    for msg in reporter.messages:
        if msg.__class__.__name__.endswith('UndefinedName'):
            assert len(msg.message_args) == 1
            names.add(msg.message_args[0])
    return names


class LevelAwareNodeTransformer(ast.NodeTransformer):
    def __init__(self):
        self.level = 0

    def visit_FunctionDef(self, node):
        self.level += 1
        self.generic_visit(node)
        self.level -= 1
        return node

    def visit_ClassDef(self, node):
        self.level += 1
        self.generic_visit(node)
        self.level -= 1
        return node


class RewriteModuleAccess(LevelAwareNodeTransformer):
    """
    Refactors attribute accesses into top-level references.
    In other words, instances of <varname>.<attr> change to <attr>.

    Any attributes that were modified are stored in `accessed_attrs`.

    Example:
        >>> from liberator.core import *
        >>> source = ub.codeblock(
        ...     '''
        ...     foo.bar = 3
        ...     foo.baz.bar = 3
        ...     biz.foo.baz.bar = 3
        ...     ''')
        >>> pt = ast.parse(source)
        >>> self = RewriteModuleAccess('foo')
        >>> orig = unparse(pt)
        >>> print(orig)
        foo.bar = 3
        foo.baz.bar = 3
        biz.foo.baz.bar = 3
        >>> self.visit(pt)
        >>> modified = unparse(pt)
        >>> print(modified)
        bar = 3
        baz.bar = 3
        biz.foo.baz.bar = 3
        >>> self.accessed_attrs
        ['bar', 'baz']
    """
    def __init__(self, modname):
        super().__init__()
        self.modname = modname
        self.accessed_attrs = []

    def update_definition(self, definition: 'Definition'):
        """
        Rewrites a definition to remote a namespace reference (i.e.  assume it
        exists in the global scope) (NOTE: we should be doing some namespace
        mangling instead)
        """
        self.visit(definition.node)
        definition._code = unparse(definition.node)
        definition._expanded = True

    def visit_Attribute(self, node):
        # print('VISIT ATTR: node = {!r}'.format(node.__dict__))
        self.generic_visit(node)
        if isinstance(node.value, ast.Name):
            if node.value.id == self.modname:
                self.accessed_attrs.append(node.attr)
                new_node = ast.Name(node.attr, node.ctx)
                old_node = node
                return ast.copy_location(new_node, old_node)
        return node


class RemoveInternalImports(ast.NodeTransformer):
    """
    Modifies an AST node to remove an internal import.
    For "well behaved code" this should leave any use of that import as an
    undefined variable.

    CommandLine:
        xdoctest -m liberator.core RemoveInternalImports

    Example:
        >>> from liberator.core import *
        >>> from liberator.core import RemoveInternalImports
        >>> from liberator.core import unparse
        >>> import ast
        >>> import copy
        >>> import sys
        >>> if sys.version_info[0:2] <= (3, 8):
        ...     import pytest
        ...     pytest.skip('Internal import removal does not work on 3.8')
        >>> source = ub.codeblock(
        ...     '''
        ...     import ubelt as ub
        ...     'a real constant'
        ...     x = 1
        ...     from ubelt import bazbiz
        ...     from ubelt.util_path import Path
        ...     from ubelt.progiter import ProgIter as Progress
        ...     from packaging.version import parse as Version
        ...     if 1:
        ...         'another real constant'
        ...         import ubelt as ub
        ...         from ubelt import bazbiz
        ...     ub.foobar()
        ...     ''')
        >>> pt = ast.parse(source)
        >>> pt1 = copy.deepcopy(pt)
        >>> node = pt.body[0]
        >>> self = RemoveInternalImports('ubelt')
        >>> self.visit(pt)
        >>> modified = unparse(pt)
        >>> print(modified)

    Ignore:
        node = pt1.body[1]
        node = pt.body[1]

        for v in pt1.body:
            print(f'{type(v)}, v.__dict__={v.__dict__}')
        print('---')
        for v in pt.body:
            print(f'{type(v)}, v.__dict__={v.__dict__}')

        pt1.body[0].lineno
        pt.body[0].lineno
    """
    def __init__(self, name):
        self.name = name
        self.level = 0
        self.rewritten_nodes = []

    def update_definition(self, definition: 'Definition'):
        """
        Rewrites a definition to remote a namespace reference (i.e.  assume it
        exists in the global scope) (NOTE: we should be doing some namespace
        mangling instead)
        """
        self.visit(definition.node)
        definition._code = unparse(definition.node)
        # FIXME: having _internal_expanded & _expanded is confusing
        definition._internal_expanded = True

    def _check_match(self, modname, name):
        parts = modname.split('.')
        matchable_modnames = []
        for i in range(1, len(parts) + 1):
            modname_prefix = '.'.join(parts[:i])
            matchable_modnames.append(modname_prefix)
        return name in matchable_modnames

    def _comment_node(self, node):
        """
        Given a node, produced a commented out version of it
        """
        line_value = unparse(node).replace('\n', '')
        prefix = '# LIBERATED(internal): '
        extra_offset = 2 + len(prefix)
        new_node = ast.Constant(
            prefix + line_value,
            # line_value,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_col_offset=node.end_col_offset + extra_offset,
            end_lineno=node.end_lineno)
        new_node._no_unparse_hack = 1
        new_node = ast.Expr(new_node)
        return new_node

    def visit_Import(self, node):
        # FIXME: multiple imports case
        new_node = None
        hacked_assigns = []
        for alias in node.names:
            modname = alias.name
            if self._check_match(modname, self.name):
                if alias.asname is not None:
                    hacked_assigns.append((alias.asname, alias.name))
                new_node = self._comment_node(node)
                self.rewritten_nodes.append(node)
        if new_node is None:
            yield node
        else:
            yield new_node
            # HACK: For aliases as-names, insert assignments
            for var, val in hacked_assigns:
                print(f'self.name={self.name}')
                print(f'val={val}')
                print(f'var={var}')
                yield ast.Assign([ast.Name(var)], ast.Name(val))

    def visit_ImportFrom(self, node):
        # FIXME: multiple imports case
        new_node = None
        hacked_assigns = []
        for alias in node.names:
            modname = node.module
            if self._check_match(modname, self.name):
                if alias.asname is not None:
                    hacked_assigns.append((alias.asname, alias.name))
                new_node = self._comment_node(node)
                self.rewritten_nodes.append(node)
        if new_node is None:
            new_node = node
        if new_node is None:
            yield node
        else:
            yield new_node
            # HACK: For aliases as-names, insert assignments
            for var, val in hacked_assigns:
                print(f'self.name={self.name}')
                print(f'val={val}')
                print(f'var={var}')
                yield ast.Assign([ast.Name(var)], ast.Name(val))


class Definition(ub.NiceRepr):
    """
    Stores a reference to some code we would like to define.

    The `node` attribute stores a potentially modified CST from which the code
    is generated.

    Used in :class:`DefinitionVisitor`.

    Example:
        >>> from liberator.core import *
        >>> from liberator.core import DefinitionVisitor
        >>> from liberator.core import RewriteModuleAccess
        >>> from liberator import core
        >>> sourcecode = ub.codeblock(
                '''
                import kwarray

                def func_with_global_ref(func):
                    kwarray.ensure_rng(1)

                def func_with_local_ref():
                    import ubelt as ub
                    return ub.Cacher
        ...     ''')
        >>> visitor = DefinitionVisitor.parse(source=sourcecode)
        >>> self = visitor.definitions['func_with_local_ref']
        >>> print(self)
        >>> rewriter = RewriteModuleAccess('ub')
        >>> rewriter.update_definition(self)
        >>> print(self.code)
        >>> # Internal import removal does not work on 3.8
        >>> import sys
        >>> if sys.version_info[0:2] >= (3, 9):
        ...     rewriter2 = RemoveInternalImports('ubelt')
        ...     rewriter2.update_definition(self)
        ...     print(self.code)

    Ignore:
        node = self.node.body[0]
    """
    def __init__(self, name, node, type=None, code=None, absname=None,
                 modpath=None, modname=None, native_modname=None,
                 minify=False):

        self.name = name
        self.node = node
        self.type = type
        self._code = code
        self.absname = absname
        self.modpath = modpath
        self.modname = modname
        self.native_modname = native_modname
        self.minify = minify
        self._expanded = False
        self._internal_expanded = False

    def _extract_code(self):
        # NOTE: the unparse variant captures decorators whereas the dynamic
        # inspect variant does not seem to do that.
        #
        # In general the inspect.getsource seems to return the same
        # formatting as the original module, but the unparse
        # is more accurate.
        try:
            if self._internal_expanded:
                raise Exception
            if self._expanded or self.type == 'Assign':
                # always use astunparse if we have expanded
                raise Exception
            if self.modname is None:
                raise Exception
            # Attempt to dynamically extract the source code because it
            # keeps formatting better.
            module = ub.import_module_from_name(self.modname)
            obj = getattr(module, self.name)
            _code = inspect.getsource(obj).strip('\n')
        except Exception:
            # Fallback on static sourcecode extraction
            # (NOTE: it should be possible to keep formatting with a bit of
            # work)
            _code = unparse(self.node).strip('\n')

        if self.minify:
            """
            TODO: good minifier
            """
            _code = remove_comments_and_docstrings2(_code)
        return _code

    @property
    def code(self):
        if self._code is None:
            self._code = self._extract_code()
        assert self._code is not None
        return self._code.strip()

    def __nice__(self):
        parts = []
        parts.append('name={!r}'.format(self.name))
        parts.append('type={!r}'.format(self.type))
        if self.absname is not None:
            parts.append('absname={!r}'.format(self.absname))
        if self.native_modname is not None:
            parts.append('native_modname={!r}'.format(self.native_modname))
        return ', '.join(parts)


class NotAPythonFile(ValueError):
    pass


class AttributeAccessVisitor(ast.NodeVisitor):
    """
    Constructs a list of all fully-specified attributes names accessed in a
    parse tree

    TODO: could use this to parse out all used attributes in current sourcecode

    Ignore:
        from liberator.core import AttributeAccessVisitor  # NOQA
        fpath = ub.expandpath('~/code/dvc/dvc/lock.py')
        sourcecode = ub.readfrom(fpath)
        pt = ast.parse(sourcecode)
        self = AttributeAccessVisitor()
        self.visit(pt)
        self.dotted_trie

        WIP: TRY TO FIX ISSUE WITH IMPORTINING SUBPACKAGES EXPLICITLY

        >>> sourcecode = ub.codeblock(
            '''
            class MyClass(foo.bar.baz):
                pass

            class MyClass3(foo.bar.baz):
                pass

            def blah():
                return foo.bar.BAZ()
            ''')
        >>> print(ub.urepr(undefined_names(sourcecode), nl=0))

        from liberator.core import DefinitionVisitor  # NOQA
        pt = ast.parse(sourcecode)

        node1 = pt.body[0].bases[0]

        visitor = AttributeAccessVisitor()
        visitor.visit(pt)
        visitor.dotted_names
    """

    def __init__(self):
        import pygtrie
        self.dotted_trie = pygtrie.StringTrie(separator='.')

    def visit_Attribute(self, node):
        curr = node
        attr_chain = []
        while isinstance(curr, ast.Attribute):
            attr_chain.append(curr.attr)
            curr = curr.value

        if isinstance(curr, ast.Name):
            attr_chain.append(curr.id)

        dotted_name = '.'.join(attr_chain[::-1])
        self.dotted_trie.setdefault(dotted_name, 0)
        self.dotted_trie[dotted_name] += 1
        # self.generic_visit(node)


class DefinitionVisitor(ast.NodeVisitor, ub.NiceRepr):
    """
    Used to search for dependencies in the original module

    References:
        https://greentreesnakes.readthedocs.io/en/latest/nodes.html

    Example:
        >>> from liberator.core import *
        >>> from liberator.core import DefinitionVisitor
        >>> from liberator import core
        >>> modpath = core.__file__
        >>> sourcecode = ub.codeblock(
        ...     '''
        ...     from ubelt.util_const import *
        ...     import a
        ...     import b
        ...     import c.d
        ...     import e.f as g
        ...     from . import h
        ...     from .i import j
        ...     from . import k, l, m
        ...     from n import o, p, q
        ...     r = 3
        ...     ''')
        >>> visitor = DefinitionVisitor.parse(source=sourcecode, modpath=modpath)
        >>> print(ub.urepr(visitor.definitions, si=1))

    Example:
        >>> from liberator.core import *
        >>> from liberator import core
        >>> modpath = core.__file__
        >>> sourcecode = ub.codeblock(
                '''
                def decor(func):
                        return func

                @decor
                def foo():
                    return 'bar'
        ...     ''')
        >>> visitor = DefinitionVisitor.parse(source=sourcecode, modpath=modpath)
        >>> print(ub.urepr(visitor.definitions, si=1))

    Example:
        >>> from liberator.core import *
        >>> from liberator.core import DefinitionVisitor
        >>> from liberator import core
        >>> modpath = core.__file__
        >>> sourcecode = ub.codeblock(
                '''
                import kwarray

                def global_import(func):
                    kwarray.ensure_rng(1)

                def nested_import():
                    import ubelt as ub
                    return ub.Cacher
        ...     ''')
        >>> visitor = DefinitionVisitor.parse(source=sourcecode, modpath=modpath)
        >>> print(ub.urepr(list(visitor.definitions), si=1))
        >>> print(ub.urepr(list(visitor.nested_definitions), si=1))

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> import mmdet
        >>> import mmdet.models
        >>> import liberator
        >>> lib = liberator.core.Liberator()
        >>> lib.add_dynamic(mmdet.models.backbones.HRNet)
        >>> print(lib.current_sourcecode())
        >>> visitor = ub.peek(lib.visitors.values())
        >>> print(ub.urepr(visitor.definitions, si=1))
        >>> d = visitor.definitions['HRNet']
        >>> print(d.code[0:1000])

    Example:
        >>> from liberator.core import *
        >>> from liberator.core import DefinitionVisitor
        >>> from liberator import core
        >>> modpath = core.__file__
        >>> # Test case global variables with type annots
        >>> sourcecode = ub.codeblock(
                '''
                GLOBAL_VAR: list = []
        ...     ''')
        >>> visitor = DefinitionVisitor.parse(source=sourcecode, modpath=modpath)
        >>> print(ub.urepr(list(visitor.definitions), si=1))
        >>> print(ub.urepr(list(visitor.nested_definitions), si=1))
    """

    def __init__(visitor, modpath=None, modname=None, module=None, pt=None,
                 logger=None):
        super(DefinitionVisitor, visitor).__init__()
        visitor.pt = pt
        visitor.modpath = modpath
        visitor.modname = modname
        visitor.module = module

        visitor.logger = logger

        import pygtrie
        visitor.definitions = pygtrie.StringTrie(separator='.')
        visitor.nested_definitions = pygtrie.StringTrie(separator='.')
        visitor.level = 0

    def __nice__(self):
        if self.modname is not None:
            return self.modname
        else:
            return "<sourcecode>"

    @classmethod
    def parse(DefinitionVisitor, source=None, modpath=None, modname=None,
              module=None, logger=None):
        if module is not None:
            if source is None:
                source = inspect.getsource(module)
            if modpath is None:
                modname = module.__file__
            if modname is None:
                modname = module.__name__

        if modpath is not None:
            if modpath.endswith('.pyc'):
                modpath = modpath.replace('.pyc', '.py')  # python 2 hack

            if isdir(modpath):
                modpath = join(modpath, '__init__.py')
            if modname is None:
                modname = ub.modpath_to_modname(modpath)

        if modpath is not None:
            if source is None:
                if not modpath.endswith(('.py', '>')):
                    raise NotAPythonFile('can only parse python files, not {}'.format(modpath))
                source = open(modpath, 'r').read()

        if source is None:
            raise ValueError('unable to derive source code')

        pt = ast.parse(source)
        visitor = DefinitionVisitor(modpath, modname, module, pt=pt,
                                    logger=logger)
        visitor.visit(pt)

        # Hack in attribute visiting
        # attr_visitor = AttributeAccessVisitor()
        # attr_visitor.visit(pt)
        # visitor.dotted_trie = attr_visitor.dotted_trie

        return visitor

    def extract_definition(visitor, name):
        """
        Given the name of a variable / class / function / moodule, extract the
        relevant lines of source code that define that structure from the
        visited module.
        """
        return visitor.definitions[name]

    def visit_Import(visitor, node):
        for d in visitor._import_definitions(node):
            if visitor.level == 0:
                visitor.definitions[d.name] = d
            else:
                visitor.nested_definitions[d.name] = d
        visitor.generic_visit(node)

    def visit_ImportFrom(visitor, node):
        for d in visitor._import_from_definition(node):
            if visitor.level == 0:
                visitor.definitions[d.name] = d
            else:
                visitor.nested_definitions[d.name] = d
        visitor.generic_visit(node)

    def _make_absname(visitor, name):
        if visitor.modname is not None:
            return visitor.modname + '.' + name
        else:
            return name

    def _common_visit_assign(visitor, node, target):
        key = getattr(target, 'id', None)
        if key is not None:
            try:
                static_val = _parse_static_node_value(node.value)
                code = '{} = {}'.format(key, ub.urepr(static_val))
            except TypeError:
                #code = unparse(node).strip('\n')
                code = None

            if visitor.logger:
                if key in visitor.definitions:
                    # OVERLOADED
                    visitor.logger.debug('OVERLOADED key = {!r}'.format(key))

            definition = Definition(
                key, node, code=code, type='Assign',
                modpath=visitor.modpath,
                modname=visitor.modname,
                absname=visitor._make_absname(key),
                native_modname=visitor.modname,
            )
            if visitor.level == 0:
                visitor.definitions[key] = definition
            # else:
            #     visitor.nested_definitions[key] = definition

    def visit_AnnAssign(visitor, node):
        if visitor.level > 0:
            return
        visitor._common_visit_assign(node, node.target)

    def visit_Assign(visitor, node):
        if visitor.level > 0:
            return
        for target in node.targets:
            visitor._common_visit_assign(node, target)

    def visit_FunctionDef(visitor, node):
        defenition = Definition(
            node.name, node, type='FunctionDef',
            modpath=visitor.modpath,
            modname=visitor.modname,
            absname=visitor._make_absname(node.name),
            native_modname=visitor.modname,
        )
        if visitor.level == 0:
            visitor.definitions[node.name] = defenition
        else:
            # visitor.nested_definitions[node.name] = defenition
            pass

        if visitor.level == 0:
            visitor.level += 1
            visitor.generic_visit(node)
            visitor.level -= 1
        else:
            visitor.generic_visit(node)
        # ast.NodeVisitor.generic_visit(visitor, node)

    def visit_ClassDef(visitor, node):
        defenition = Definition(
            node.name, node, type='ClassDef',
            modpath=visitor.modpath,
            modname=visitor.modname,
            absname=visitor._make_absname(node.name),
            native_modname=visitor.modname,
        )
        if visitor.level == 0:
            visitor.definitions[node.name] = defenition
        else:
            # visitor.nested_definitions[node.name] = defenition
            pass

        if visitor.level == 0:
            visitor.level += 1
            visitor.generic_visit(node)
            visitor.level -= 1
        else:
            visitor.generic_visit(node)

        # # Ignore any non-top-level imports
        # if not visitor.level == 0:
        #     # ast.NodeVisitor.generic_visit(visitor, node)

    def _import_definitions(visitor, node):
        for alias in node.names:
            varname = alias.asname or alias.name
            if alias.asname:
                line = 'import {} as {}'.format(alias.name, alias.asname)
            else:
                line = 'import {}'.format(alias.name)
            absname = alias.name
            yield Definition(varname, node, code=line,
                             absname=absname,
                             native_modname=absname,
                             modpath=visitor.modpath,
                             modname=visitor.modname,
                             type='Import')

    def _import_from_definition(visitor, node):
        """
        Ignore:
            from liberator.core import *
            visitor = DefinitionVisitor.parse(module=module)
            print('visitor.definitions = {}'.format(ub.urepr(visitor.definitions, sv=1)))
        """
        if node.level:
            # Handle relative imports
            if visitor.modpath is not None:
                try:
                    rel_modpath = ub.split_modpath(abspath(visitor.modpath))[1]
                except ValueError:
                    warnings.warn('modpath={} does not exist'.format(visitor.modpath))
                    rel_modpath = basename(abspath(visitor.modpath))
                modparts = rel_modpath.replace('\\', '/').split('/')
                parts = modparts[:-node.level]
                prefix = '.'.join(parts)
                if node.module:
                    prefix = prefix + '.'
            else:
                warnings.warn('Unable to rectify absolute import')
                prefix = '.' * node.level
        else:
            prefix = ''

        if node.module is not None:
            abs_modname = prefix + node.module
        else:
            abs_modname = prefix

        for alias in node.names:
            varname = alias.asname or alias.name
            if alias.asname:
                line = 'from {} import {} as {}'.format(abs_modname, alias.name, alias.asname)
            else:
                line = 'from {} import {}'.format(abs_modname, alias.name)
            absname = abs_modname + '.' + alias.name
            if varname == '*':
                # HACK
                abs_modpath = ub.modname_to_modpath(abs_modname)
                star_visitor = DefinitionVisitor.parse(
                    modpath=abs_modpath, logger=visitor.logger)
                for d in star_visitor.definitions.values():
                    if not d.name.startswith('_'):
                        yield d
            else:
                yield Definition(varname, node, code=line, absname=absname,
                                 modpath=visitor.modpath,
                                 modname=visitor.modname,
                                 native_modname=abs_modname,
                                 type='ImportFrom')


def _closefile(fpath, modnames):
    """
    An api to remove dependencies from code by "closing" them.

    CommandLine:
        xdoctest -m ~/code/liberator/core.py _closefile
        xdoctest -m liberator.core _closefile --fpath=~/code/boltons/tests/test_cmdutils.py --modnames=ubelt,
        xdoctest -m liberator.core _closefile --fpath=~/code/dvc/dvc/updater.py --modnames=dvc,

    Example:
        >>> # SCRIPT
        >>> # ENTRYPOINT
        >>> import scriptconfig as scfg
        >>> config = scfg.quick_cli({
        >>>     'fpath': scfg.Path(None),
        >>>     'modnames': scfg.Value([]),
        >>> })
        >>> #fpath = config['fpath'] = ub.expandpath('~/code/boltons/tests/test_cmdutils.py')
        >>> #modnames = config['modnames'] = ['ubelt']
        >>> _closefile(**config)
    """
    from xdoctest import static_analysis as static
    modpath = fpath
    expand_names = modnames
    source = ub.Path(fpath).read_text()
    calldefs = static.parse_calldefs(source, fpath)
    calldefs.pop('__doc__', None)

    lib = Liberator()
    for key in calldefs.keys():
        lib.add_static(key, modpath)
    lib.expand(expand_names)
    #print(ub.urepr(lib.body_defs, si=1))
    print(lib.current_sourcecode())


class Closer(Liberator):
    """
    Deprecated in favor of :class:`Liberator`.

    The original name of the Liberator class was called Closer. Exposing this
    for backwards compatibility.
    """
    pass


def remove_comments_and_docstrings(source):
    r"""
    Args:
        source (str): uft8 text of source code

    Returns:
        str: out: the source with comments and docstrings removed.

    References:
        https://stackoverflow.com/questions/1769332/remove-comments-docstrings

    Example:
        >>> source = ub.codeblock(
            '''
            def foo():
                'The spaces before this docstring are tokenize.INDENT'
                test = [
                    'The spaces before this string do not get a token'
                ]
            ''')
        >>> out = remove_comments_and_docstrings(source)
        >>> want = ub.codeblock(
            '''
            def foo():
                pass
                test = [
                    'The spaces before this string do not get a token'
                ]''').splitlines()
        >>> got = [o.rstrip() for o in out.splitlines()]
        >>> assert got == want


        >>> source = ub.codeblock(
            '''
            def foo():
                " docstring "
            ''')
        >>> out = remove_comments_and_docstrings(source)
        >>> print(out)
        >>> source = ub.codeblock(
            '''
            class foo():
                r{qqq}
                docstring
                {qqq}
            ''').format(qqq='"' * 3)
        >>> out = remove_comments_and_docstrings(source)
        >>> print(out)

    """
    import tokenize
    source = str(source)
    io_obj = io.StringIO(source)
    output_parts = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        # ltext = tok[4]
        # The following two conditionals preserve indentation.
        # This is necessary because we're not using tokenize.untokenize()
        # (because it spits out code with copious amounts of oddly-placed
        # whitespace).
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            output_parts.append((' ' * (start_col - last_col)))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                # This is likely a docstring; double-check we're not inside an
                # operator:
                if prev_toktype != tokenize.NEWLINE:
                    # Note regarding NEWLINE vs NL: The tokenize module
                    # differentiates between newlines that start a new statement
                    # and newlines inside of operators such as parens, brackes,
                    # and curly braces.  Newlines inside of operators are
                    # NEWLINE and newlines that start new code are NL.
                    # Catch whole-module docstrings:
                    if start_col > 0:
                        # Unlabelled indentation means we're inside an operator
                        output_parts.append(token_string)
                    # Note regarding the INDENT token: The tokenize module does
                    # not label indentation inside of an operator (parens,
                    # brackets, and curly braces) as actual indentation.
            else:
                # NOTE: simply removing docstrings may create invalid code
                # in cases where the only body is a docstring (e.g. a
                # custom exception). Insert a pass to prevent this.  It
                # would be nice to detect when this is necessary.
                output_parts.append('pass')
        else:
            output_parts.append(token_string)
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = ''.join(output_parts)
    return out


def remove_comments_and_docstrings2(source):
    # https://stackoverflow.com/a/62074206/887074
    import io, tokenize, re  # NOQA
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]  # NOQA
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(line for line in out.splitlines() if line.strip())
    return out
