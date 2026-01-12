# -*- coding: utf-8 -*-
import numpy as np
import ubelt as ub
import re
import os


class SupressPrint():
    """
    Temporarily replace the print function in a module with a noop

    Args:
        *mods: the modules to disable print in
        **kw: only accepts "enabled"
            enabled (bool, default=True): enables or disables this context
    """
    def __init__(self, *mods, **kw):
        enabled = kw.get('enabled', True)
        self.mods = mods
        self.enabled = enabled
        self.oldprints = {}

    def __enter__(self):
        if self.enabled:
            for mod in self.mods:
                oldprint = getattr(self.mods, 'print', print)
                self.oldprints[mod] = oldprint
                mod.print = lambda *args, **kw: None
    def __exit__(self, a, b, c):
        if self.enabled:
            for mod in self.mods:
                mod.print = self.oldprints[mod]


class FlatIndexer(ub.NiceRepr):
    """
    Creates a flat "view" of a jagged nested indexable object.
    Only supports one offset level.

    TODO:
        - [ ] Move to kwarray

    Args:
        lens (list): a list of the lengths of the nested objects.

    Doctest:
        >>> self = FlatIndexer([1, 2, 3])
        >>> len(self)
        >>> self.unravel(4)
        >>> self.ravel(2, 1)
    """
    def __init__(self, lens, cums=None):
        self.lens = lens
        if cums is None:
            self.cums = np.cumsum(lens)
        else:
            self.cums = cums

    def concat(self, other):
        """
        >>> self = FlatIndexer([1, 2, 3])
        >>> self = self.concat(self).concat(self)
        >>> len(self)
        """
        new_lens = self.lens + other.lens
        new_cums = np.concatenate([self.cums, other.cums + self.cums[-1]], axis=0)
        new = self.__class__(new_lens, new_cums)
        return new

    def subslice(self, start, stop):
        """
        >>> self = FlatIndexer([3, 7, 9, 4, 5] + [3] * 50)
        >>> start = 4
        >>> stop = 150
        >>> self.subslice(start, stop)
        >>> self.subslice(0, 10).cums
        """
        outer1, inner1  = self.unravel(start)
        outer2, inner2  = self.unravel(stop)
        return self._subslice(outer1, outer2, inner1, inner2)

    def _subslice(self, outer1, outer2, inner1, inner2):
        inner2 = min(self.lens[outer2], inner2)
        if outer1 == outer2:
            new_lens = [inner2 - inner1]
            new_cums = np.array(new_lens)
        else:
            first = [self.lens[outer1] - inner1]
            inner = self.lens[outer1 + 1:outer2]
            last = [inner2]

            new_lens = first + inner + last
            # Oddly, this is faster than just redoing the cumsum
            # or is it now that we added a copy?
            new_cums = self.cums[outer1:outer2 + 1].copy()
            new_cums -= (new_cums[0] - first[0])
            new_cums[-1] = new_cums[-2] + inner2

            if new_lens[-1] == 0:
                new_lens = new_lens[:-1]
                new_cums = new_cums[:-1]
        new = self.__class__(new_lens, new_cums)
        return new

    @classmethod
    def fromlist(cls, items):
        lens = list(map(len, items))
        return cls(lens)

    def __len__(self):
        return self.cums[-1] if len(self.cums) else 0

    def unravel(self, index):
        """
        Args:
            index : raveled index

        Returns:
            Tuple[int, int]: outer and inner indices

        Example:
            >>> self = FlatIndexer([1, 1])
            >>> index = 2
            >>> self.unravel(2)
        """
        found = np.where(self.cums > index)[0]
        # Keep indexing past the end of the last bucket for slicing
        outer = found[0] if len(found) else len(self.cums) - 1
        base = self.cums[outer] - self.lens[outer]
        inner = index - base
        return (outer, inner)

    def ravel(self, outer, inner):
        """
        Args:
            outer: index into outer list
            inner: index into the list referenced by outer

        Returns:
            index: the raveled index
        """
        base = self.cums[outer] - self.lens[outer]
        return base + inner


def strip_ansi(text):
    r"""
    Removes all ansi directives from the string.

    References:
        http://stackoverflow.com/questions/14693701/remove-ansi
        https://stackoverflow.com/questions/13506033/filtering-out-ansi-escape-sequences

    Examples:
        >>> line = '\t\u001b[0;35mBlabla\u001b[0m     \u001b[0;36m172.18.0.2\u001b[0m'
        >>> escaped_line = strip_ansi(line)
        >>> assert escaped_line == '\tBlabla     172.18.0.2'
    """
    # ansi_escape1 = re.compile(r'\x1b[^m]*m')
    # text = ansi_escape1.sub('', text)
    # ansi_escape2 = re.compile(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?')
    ansi_escape3 = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]', flags=re.IGNORECASE)
    if os.name != 'nt':
        text = ansi_escape3.sub('', text)
    return text


def align(text, character='=', replchar=None, pos=0):
    r"""
    Left justifies text on the left side of character

    Args:
        text (str): text to align
        character (str): character to align at
        replchar (str): replacement character (default=None)

    Returns:
        str: new_text

    Example:
        >>> character = '='
        >>> text = 'a = b=\none = two\nthree = fish\n'
        >>> print(text)
        >>> result = (align(text, '='))
        >>> print(result)
        a     = b=
        one   = two
        three = fish
    """
    line_list = text.splitlines()
    new_lines = align_lines(line_list, character, replchar, pos=pos)
    new_text = '\n'.join(new_lines)
    return new_text


def align_lines(line_list, character='=', replchar=None, pos=0):
    r"""
    Left justifies text on the left side of character

    align_lines

    TODO:
        clean up and move to ubelt?

    Args:
        line_list (list of strs):
        character (str):
        pos (int or list or None): does one alignment for all chars beyond this
            column position. If pos is None, then all chars are aligned.

    Returns:
        list: new_lines

    Example:
        >>> line_list = 'a = b\none = two\nthree = fish'.split('\n')
        >>> character = '='
        >>> new_lines = align_lines(line_list, character)
        >>> result = ('\n'.join(new_lines))
        >>> print(result)
        a     = b
        one   = two
        three = fish

    Example:
        >>> line_list = 'foofish:\n    a = b\n    one    = two\n    three    = fish'.split('\n')
        >>> character = '='
        >>> new_lines = align_lines(line_list, character)
        >>> result = ('\n'.join(new_lines))
        >>> print(result)
        foofish:
            a        = b
            one      = two
            three    = fish

    Example:
        >>> import ubelt as ub
        >>> character = ':'
        >>> text = ub.codeblock('''
            {'max': '1970/01/01 02:30:13',
             'mean': '1970/01/01 01:10:15',
             'min': '1970/01/01 00:01:41',
             'range': '2:28:32',
             'std': '1:13:57',}''').split('\n')
        >>> new_lines = align_lines(text, ':', ' :')
        >>> result = '\n'.join(new_lines)
        >>> print(result)
        {'max'   : '1970/01/01 02:30:13',
         'mean'  : '1970/01/01 01:10:15',
         'min'   : '1970/01/01 00:01:41',
         'range' : '2:28:32',
         'std'   : '1:13:57',}

    Example:
        >>> line_list = 'foofish:\n a = b = c\n one = two = three\nthree=4= fish'.split('\n')
        >>> character = '='
        >>> # align the second occurence of a character
        >>> new_lines = align_lines(line_list, character, pos=None)
        >>> print(('\n'.join(line_list)))
        >>> result = ('\n'.join(new_lines))
        >>> print(result)
        foofish:
         a   = b   = c
         one = two = three
        three=4    = fish
    """

    # FIXME: continue to fix ansi
    if pos is None:
        # Align all occurences
        num_pos = max([line.count(character) for line in line_list])
        pos = list(range(num_pos))

    # Allow multiple alignments
    if isinstance(pos, list):
        pos_list = pos
        # recursive calls
        new_lines = line_list
        for pos in pos_list:
            new_lines = align_lines(new_lines, character=character,
                                    replchar=replchar, pos=pos)
        return new_lines

    # base case
    if replchar is None:
        replchar = character

    # the pos-th character to align
    lpos = pos
    rpos = lpos + 1

    tup_list = [line.split(character) for line in line_list]

    handle_ansi = True
    if handle_ansi:
        # Remove ansi from length calculation
        # References: http://stackoverflow.com/questions/14693701remove-ansi
        ansi_escape = re.compile(r'\x1b[^m]*m')

    # Find how much padding is needed
    maxlen = 0
    for tup in tup_list:
        if len(tup) >= rpos + 1:
            if handle_ansi:
                tup = [ansi_escape.sub('', x) for x in tup]
            left_lenlist = list(map(len, tup[0:rpos]))
            left_len = sum(left_lenlist) + lpos * len(replchar)
            maxlen = max(maxlen, left_len)

    # Pad each line to align the pos-th occurence of the chosen character
    new_lines = []
    for tup in tup_list:
        if len(tup) >= rpos + 1:
            lhs = character.join(tup[0:rpos])
            rhs = character.join(tup[rpos:])
            # pad the new line with requested justification
            newline = lhs.ljust(maxlen) + replchar + rhs
            new_lines.append(newline)
        else:
            new_lines.append(replchar.join(tup))
    return new_lines
