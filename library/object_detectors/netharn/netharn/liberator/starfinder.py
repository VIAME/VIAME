

def find_import_stars(text):
    """
    Args:
        text (str): the python code to refactor

    Example:
        >>> # xdoctest: +REQUIRES(module:parso)
        >>> from liberator.starfinder import *  # NOQA
        >>> import ubelt as ub
        >>> text = ub.codeblock(
        >>>     '''
        >>>     import dis as dat
        >>>     from io import *
        >>>     from a.b import *
        >>>     from textwrap import *  # NOQA
        >>>     x = StringIO
        >>>     y = dedent
        >>>     ''')
        >>> final_text = find_import_stars(text)
        >>> print('----')
        >>> print('Text')
        >>> print('----')
        >>> print(ub.highlight_code(text))
        >>> print('----------')
        >>> print('Final Text')
        >>> print('----------')
        >>> print(ub.highlight_code(final_text))
    """
    import parso
    import ubelt as ub
    import_star_infos = []
    mod = parso.parse(text)
    for node in mod.iter_imports():
        if len(node.children) > 1:
            a, b = node.children[0:2]
            y, z = node.children[-2:]
            parts_ayz = [a.get_code().strip(),
                         y.get_code().strip(),
                         z.get_code().strip()]
            flag = parts_ayz == ['from', 'import', '*']
            if flag:
                # This is an import * node we want to replace it
                # Use this to figure out stuff about the module
                modname = b.get_code().strip()
                modpath = ub.modname_to_modpath(modname)
                import_star_infos.append({
                    'node': node,
                    'modname': modname,
                    'modpath': modpath,
                })

    from .core import undefined_names
    lines = text.split('\n')
    for info in reversed(import_star_infos):
        node = info['node']
        s = node.start_pos[0] - 1
        t = node.end_pos[0]
        del lines[s:t]

    new_code = '\n'.join(lines)

    names = undefined_names(new_code)
    # Now we need to associate which undefined name comes from which import *
    new_lines = []
    associated = names  # hack only works for one
    for info in import_star_infos:
        modname = info['modname']
        if associated:
            associated_part = ', '.join(associated)
            new_lines.append(f'from {modname} import {associated_part}')
        associated = None
    # But if there is just one, then we can skip this check in some cases.
    final_text = '\n'.join(new_lines) + '\n' + new_code
    return final_text
