"""
Export component of the Pytorch exporter.

This is the code that simply exports the model toplogy via code

Uses static analysis to export relevant code that defines the model topology
into a stanadlone file. As long as your model definition is indepenent of your
training code, then the exported file can be passed around in a similar way to
a caffe prototext file.

TODO:
    - [ ]: Look into: https://www.reddit.com/r/MachineLearning/comments/a856oe/d_pytorch_10_deployment_pipeline/ec9w94c/

    >>> from torchvision.models import densenet
    >>> import torch
    >>> model = densenet.DenseNet(growth_rate=16).eval()
    >>> traced = torch.jit.trace(model, example_inputs=(torch.randn(2, 3, 224, 224), ))
    >>> traced.save("densenet.pt")
    >>> model_ = torch.jit.load("densenet.pt")


CommandLine:
    xdoctest -m torch_liberator.exporter export_model_code
    xdoctest -m torch_liberator.exporter all
"""
import ast
import re
import hashlib
import io
import pickle
import tokenize
import ubelt as ub
import warnings
from os.path import join
from viame.pytorch.netharn import liberator

__all__ = ['export_model_code']


__pt_export_version__ = '0.7.0'


def export_model_code(dpath, model, initkw=None, export_modules=[]):
    """
    Exports the class used to define a pytorch model as a new python module.

    Exports the minimum amount of code needed to make a self-contained Python
    module defining the pytorch model class. This exports the actual source
    code. The advantage of using this over pickle is that the original code can
    change arbitrarilly because all dependencies on the original code are
    removed in the exported code.

    Args:
        dpath (str): directory to dump the model
        model (tuple or type or object): class or class instance (e.g. torch.nn.Module)
        name (str): name to use for the file (defaults to the classname)
        initkw (dict): if specified, creates the function `make`, which
            initializes the network with the specific arguments.
        export_modules (List[str]): A list of modules that the exported code
            should not depend on. Any code referenced from these modules will
            be statically extracted and copied into the model definition.
            Note that this feature is experimental.

    Returns:
        str: static_modpath: path to the saved model file.
            While you could put the output path in your PYTHONPATH, it is best
            to use `ub.import_module_from_path` to "load" the model instead.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchvision)
        >>> from torch_liberator.exporter import export_model_code
        >>> from torchvision.models import densenet
        >>> import torchvision
        >>> from os.path import basename
        >>> initkw = {'growth_rate': 16}
        >>> model = densenet.DenseNet(**initkw)
        >>> dpath = ub.ensure_app_cache_dir('torch_liberator/tests')
        >>> static_modpath = export_model_code(dpath, model, initkw)
        >>> print('static_modpath = {!r}'.format(static_modpath))
        ...
        >>> mod_fname = (basename(static_modpath))
        >>> print('mod_fname = {!r}'.format(mod_fname))
        >>> if torchvision.__version__ == '0.2.2':
        >>>     assert mod_fname == 'DenseNet_256629.py', 'got={}'.format(mod_fname)
        >>> # now the module can be loaded
        >>> module = ub.import_module_from_path(static_modpath)
        >>> loaded = module.make()
        >>> assert model.features.denseblock1.denselayer1.conv2.out_channels == 16
        >>> assert loaded.features.denseblock1.denselayer1.conv2.out_channels == 16
        >>> assert model is not loaded
    """
    if isinstance(model, type):
        model_class = model
    else:
        model_class = model.__class__
    classname = model_class.__name__

    if initkw is None:
        raise NotImplementedError(
            'ERROR: The params passed to the model __init__ must be available')
        footer = ''
    else:
        # First see if we can get away with a simple encoding of initkw
        try:
            # Do not use repr. The text produced is non-deterministic for
            # dictionaries. Instead, use ub.urepr with repr2's old dict
            # ordering, which is deterministic.
            #
            # Do not "modernize" this to a plain urepr or to urepr(sort=True).
            # This text is written into the generated topology as the model's
            # initkw and is hashed for the deploy name, and urepr(sort=True)
            # sorts list and tuple *contents* as well as dict keys -- it would
            # silently permute 'classes', remapping every label index.
            # _dict_sort_behavior='old' is exactly what ub.repr2 passed, so
            # output is unchanged; if ubelt drops it the failure is a loud
            # TypeError rather than quietly reordered labels.
            init_text = ub.urepr(initkw, nl=1, _dict_sort_behavior='old')
            eval(init_text, {})
            init_code = ub.codeblock(
                'initkw = {}'
            ).format(init_text)
        except Exception:
            # fallback to pickle
            warnings.warn('Initialization params might not be serialized '
                          'deterministically')
            init_bytes = repr(pickle.dumps(initkw, protocol=0))
            init_code = ub.codeblock(
                '''
                import pickle
                initkw = pickle.loads({})
                '''
            ).format(init_bytes)
        init_code = ub.indent(init_code).lstrip()
        # create a function to instanciate the class
        footer = '\n\n' + ub.codeblock(
            '''
            __pt_export_version__ = '{__pt_export_version__}'


            def get_initkw():
                """ creates an instance of the model """
                {init_code}
                return initkw


            def get_model_cls():
                model_cls = {classname}
                return model_cls


            def make():
                """ creates an instance of the model """
                initkw = get_initkw()
                model_cls = get_model_cls()
                model = model_cls(**initkw)
                return model
            '''
        ).format(classname=classname, init_code=init_code,
                 __pt_export_version__=__pt_export_version__)

        # TODO: assert that the name "make" is not used in the model body

    try:
        lib = liberator.Liberator()
    except AttributeError:
        from viame.pytorch.netharn.liberator.core import Closer
        lib = Closer()

    obj = model_class
    expand_names = export_modules
    import sys
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

    body = closed_sourcecode

    body_footer = body + footer + '\n'
    # dont need to hash the header, because comments are removed anyway

    # with open('debug-closer.py', 'w') as file:
    #     file.write(body_footer)
    hashid = hash_code(body_footer)

    header = ub.codeblock(
        '''
        """
        This module was autogenerated by torch_liberator/export/exporter.py
        original_module={}
        classname={}
        timestamp={}
        hashid={}
        """
        ''').format(model_class.__module__, classname, ub.timestamp(), hashid)

    sourcecode = header + '\n' + body_footer

    static_modname = classname + '_' + hashid[0:6]
    static_modpath = join(dpath, static_modname + '.py')
    with open(static_modpath, 'w') as file:
        file.write(sourcecode)
    return static_modpath


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


def hash_code(sourcecode):
    r"""
    Hashes source code text, but tries to normalize things like whitespace and
    comments, so very minor changes wont change the hash.

    Args:
        source (str): uft8 text of source code

    Returns:
        str: hashid: 128 character (512 byte) hash of the normalized input

    Notes:
        The return value of this function is based on the AST parse tree, which
        might change between different version of Python. However, within the
        same version of Python, the results should be consistent.

    CommandLine:
        xdoctest -m /home/joncrall/code/torch_liberator/torch_liberator/export/exporter.py hash_code

    Example:
        >>> hashid1 = (hash_code('x = 1')[0:8])
        >>> hashid2 = (hash_code('x=1 # comments and spaces dont matter')[0:8])
        >>> hashid3 = (hash_code('\nx=1')[0:8])
        >>> assert ub.allsame([hashid1, hashid2, hashid3])
        >>> hashid4 = hash_code('x=2')[0:8]
        >>> assert hashid1 != hashid4
    """
    # Strip docstrings before making a parse tree
    sourcecode = str(sourcecode)

    stripped = remove_comments_and_docstrings(sourcecode)

    # Also remove pytorch_export version info (not sure if correct?)
    stripped = re.sub('__pt_export_version__ = .*', '', stripped)

    parse_tree = ast.parse(stripped)
    # hashing the parse tree will normalize for a lot possible small changes
    ast_dump = ast.dump(parse_tree)

    hasher = hashlib.sha512()
    hasher.update(ast_dump.encode('utf8'))
    hashid = hasher.hexdigest()
    return hashid


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m torch_liberator.exporter
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
