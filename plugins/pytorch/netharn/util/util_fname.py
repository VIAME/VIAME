# -*- coding: utf-8 -*-
"""
NOTE: THIS FILE IS DEPRECATED

Processing for filenames. The logic is relatively hacky.

pip install pygtrie
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import commonprefix, isdir, dirname
import numpy as np  # NOQA


def shortest_unique_prefixes(items, sep=None, allow_simple=True, min_length=0, allow_end=False):
    r"""
    The shortest unique prefix algorithm.

    Args:
        items (list of str): returned prefixes will be unique wrt this set
        sep (str): if specified, all characters between separators are treated
            as a single symbol. Makes the algo much faster.
        allow_simple (bool): if True tries to construct a simple feasible
            solution before resorting to the optimal trie algorithm.
        min_length (int): minimum length each prefix can be
        allow_end (bool): if True allows for string terminators to be
            considered in the prefix

    Returns:
        list of str: a prefix for each item that uniquely identifies it
           wrt to the original items.

    References:
        http://www.geeksforgeeks.org/find-all-shortest-unique-prefixes-to-represent-each-word-in-a-given-list/
        https://github.com/Briaares/InterviewBit/blob/master/Level6/Shortest%20Unique%20Prefix.cpp

    Requires:
        pip install pygtrie

    Example:
        >>> # xdoctest: +REQUIRES(--pygtrie)
        >>> items = ["zebra", "dog", "duck", "dove"]
        >>> shortest_unique_prefixes(items)
        ['z', 'dog', 'du', 'dov']

    Timeing:
        >>> # DISABLE_DOCTEST
        >>> # make numbers larger to stress test
        >>> # L = max length of a string, N = number of strings,
        >>> # C = smallest gaurenteed common length
        >>> # (the setting N=10000, L=100, C=20 is feasible we are good)
        >>> import ubelt as ub
        >>> import random
        >>> def make_data(N, L, C):
        >>>     rng = random.Random(0)
        >>>     return [''.join(['a' if i < C else chr(rng.randint(97, 122))
        >>>                      for i in range(L)]) for _ in range(N)]
        >>> items = make_data(N=1000, L=10, C=0)
        >>> ub.Timerit(3).call(shortest_unique_prefixes, items).print()
        Timed for: 3 loops, best of 3
            time per loop: best=24.54 ms, mean=24.54 ± 0.0 ms
        >>> items = make_data(N=1000, L=100, C=0)
        >>> ub.Timerit(3).call(shortest_unique_prefixes, items).print()
        Timed for: 3 loops, best of 3
            time per loop: best=155.4 ms, mean=155.4 ± 0.0 ms
        >>> items = make_data(N=1000, L=100, C=70)
        >>> ub.Timerit(3).call(shortest_unique_prefixes, items).print()
        Timed for: 3 loops, best of 3
            time per loop: best=232.8 ms, mean=232.8 ± 0.0 ms
        >>> items = make_data(N=10000, L=250, C=20)
        >>> ub.Timerit(3).call(shortest_unique_prefixes, items).print()
        Timed for: 3 loops, best of 3
            time per loop: best=4.063 s, mean=4.063 ± 0.0 s
    """
    import pygtrie
    if len(set(items)) != len(items):
        raise ValueError('inputs must be unique')

    # construct trie
    if sep is None:
        trie = pygtrie.CharTrie.fromkeys(items, value=0)
    else:
        # In some simple cases we can avoid constructing a trie
        if allow_simple:
            tokens = [item.split(sep) for item in items]
            simple_solution = [t[0] for t in tokens]
            if len(simple_solution) == len(set(simple_solution)):
                return simple_solution
            for i in range(2, 10):
                # print('return simple solution at i = {!r}'.format(i))
                simple_solution = ['-'.join(t[:i]) for t in tokens]
                if len(simple_solution) == len(set(simple_solution)):
                    return simple_solution

        trie = pygtrie.StringTrie.fromkeys(items, value=0, separator=sep)

    # Set the value (frequency) of all nodes to zero.
    for node in _trie_iternodes(trie):
        node.value = 0

    # For each item trace its path and increment frequencies
    for item in items:
        final_node, trace = trie._get_node(item)
        for key, node in trace:
            node.value += 1

    # if not isinstance(node.value, int):
    #     node.value = 0

    # Query for the first prefix with frequency 1 for each item.
    # This is the shortest unique prefix over all items.
    unique = []
    for item in items:
        freq = None
        for prefix, freq in trie.prefixes(item):
            if freq == 1 and len(prefix) >= min_length:
                break
        if not allow_end:
            assert freq == 1, 'item={} has no unique prefix. freq={}'.format(item, freq)
        # print('items = {!r}'.format(items))
        unique.append(prefix)
    return unique


def _trie_iternodes(self):
    """
    Generates all nodes in the trie

    # Hack into the internal structure and insert frequencies at each node
    """
    from collections import deque
    stack = deque([[self._root]])
    while stack:
        for node in stack.pop():
            yield node
            try:
                # only works in pygtrie-2.2 broken in pygtrie-2.3.2
                stack.append(node.children.values())
            except AttributeError:
                stack.append([v for k, v in node.children.iteritems()])


def shortest_unique_suffixes(items, sep=None):
    r"""
    CommandLine:
        xdoctest -m netharn.util.util_fname shortest_unique_suffixes

    Example:
        >>> # xdoctest: +REQUIRES(--pygtrie)
        >>> items = ["zebra", "dog", "duck", "dove"]
        >>> shortest_unique_suffixes(items)
        ['a', 'g', 'k', 'e']

    Example:
        >>> # xdoctest: +REQUIRES(--pygtrie)
        >>> items = ["aa/bb/cc", "aa/bb/bc", "aa/bb/dc", "aa/cc/cc"]
        >>> shortest_unique_suffixes(items)
    """
    snoitpo = [p[::-1] for p in items]
    sexiffus = shortest_unique_prefixes(snoitpo, sep=sep)
    suffixes = [s[::-1] for s in sexiffus]
    return suffixes


def dumpsafe(paths, repl='<sl>'):
    """
    enforces that filenames will not conflict.
    Removes common the common prefix, and replaces slashes with <sl>

    Ignore:
        >>> # xdoctest: +REQUIRES(--pygtrie)
        >>> paths = ['foo/{:04d}/{:04d}'.format(i, j) for i in range(2) for j in range(20)]
        >>> list(dumpsafe(paths, '-'))
    """
    common_pref = commonprefix(paths)
    if not isdir(common_pref):
        im_pref = dirname(common_pref)
        if common_pref[len(im_pref):len(im_pref) + 1] == '/':
            im_pref += '/'
        elif common_pref[len(im_pref):len(im_pref) + 1] == '\\':
            im_pref += '\\'
    else:
        im_pref = common_pref

    start = len(im_pref)
    dump_paths = (
        p[start:].replace('/', repl).replace('\\', repl)  # faster
        # relpath(p, im_pref).replace('/', repl).replace('\\', repl)
        for p in paths
    )
    return dump_paths


def _fast_name_we(fname):
    # Assume that extensions are no more than 7 chars for extra speed
    pos = fname.rfind('.', -7)
    return fname if pos == -1 else fname[:pos]


def _fast_basename_we(fname):
    slashpos = fname.rfind('/')
    base = fname if slashpos == -1 else fname[slashpos + 1:]
    pos = base.rfind('.', -slashpos)
    base_we = base if pos == -1 else base[:pos]
    return base_we


def _safepaths(paths):
    r"""
    Ignore:
        x = '/home/local/KHQ/jon.crall/code/netharn/netharn/live/urban_train.py'
        import re
        %timeit splitext(x.replace('<sl>', '-').replace('_', '-'))[0]
        %timeit splitext(re.sub('<sl>|_', '-', x))
        %timeit x[:x.rfind('.')].replace('<sl>', '-').replace('_', '-')
        %timeit _fast_name_we(x)
        %timeit x[:x.rfind('.')]

        >>> paths = ['foo/{:04d}/{:04d}'.format(i, j) for i in range(2) for j in range(20)]
        >>> _safepaths(paths)
    """
    safe_paths = [
        # faster than splitext
        _fast_name_we(x).replace('_', '-').replace('<sl>', '-')
        # splitext(x.replace('<sl>', '-').replace('_', '-'))[0]
        for x in dumpsafe(paths, repl='-')
    ]
    return safe_paths


def align_paths(paths1, paths2):
    r"""
    return path2 in the order of path1

    This function will only work where file types (i.e. image / groundtruth)
    are specified by EITHER a path prefix XOR a path suffix (note this is an
    exclusive or. do not mix prefixes and suffixes), either as part of a
    filename or parent directory. In the case of a filename it is assumped this
    "type identifier" is separated from the rest of the path by an underscore
    or hyphen.

    paths1, paths2 = gt_paths, pred_paths

    Doctest:
        >>> # xdoc: +REQUIRES(--pygtrie)
        >>> def test_gt_arrangements(paths1, paths2, paths2_):
        >>>     # no matter what order paths2_ comes in, it should align with the groundtruth
        >>>     assert align_paths(paths1, paths2_) == paths2
        >>>     assert align_paths(paths1[::-1], paths2_) == paths2[::-1]
        >>>     assert align_paths(paths1[0::2] + paths1[1::2], paths2_) == paths2[0::2] + paths2[1::2]
        >>>     sortx = np.arange(len(paths1))
        >>>     np.random.shuffle(sortx)
        >>>     assert align_paths(list(np.take(paths1, sortx)), paths2_) == list(np.take(paths2, sortx))
        >>> #
        >>> def test_input_arrangements(paths1, paths2):
        >>>     paths2_ = list(paths2)
        >>>     test_gt_arrangements(paths1, paths2, paths2_)
        >>>     test_gt_arrangements(paths1, paths2, paths2_[::-1])
        >>>     np.random.shuffle(paths2_)
        >>>     test_gt_arrangements(paths1, paths2, paths2_)
        >>> paths1 = ['foo/{:04d}/{:04d}'.format(i, j) for i in range(2) for j in range(20)]
        >>> paths2 = ['bar/{:04d}/{:04d}'.format(i, j) for i in range(2) for j in range(20)]
        >>> test_input_arrangements(paths1, paths2)
        >>> paths1 = ['foo/{:04d}/{:04d}'.format(i, j) for i in range(2) for j in range(20)]
        >>> paths2 = ['bar<sl>{:04d}<sl>{:04d}'.format(i, j) for i in range(2) for j in range(20)]
        >>> test_input_arrangements(paths1, paths2)

    Speed:
        >>> import ubelt as ub
        >>> paths1 = [ub.expandpath('~/foo/{:04d}/{:04d}').format(i, j) for i in range(2) for j in range(10000)]
        >>> paths2 = [ub.expandpath('~/bar/{:04d}/{:04d}').format(i, j) for i in range(2) for j in range(10000)]
        >>> np.random.shuffle(paths2)
        >>> aligned = align_paths(paths1, paths2)

        items = [p[::-1] for p in _safepaths(paths1)]

    Ignore:
        >>> # pathological case (can we support this?)
        >>> aligned = [
        >>>     ('ims/aaa.png', 'gts/aaa.png'),
        >>>     ('ims/bbb.png', 'gts/bbb.png'),
        >>>     ('ims/ccc.png', 'gts/ccc.png'),
        >>>     # ---
        >>>     ('aaa/im.png', 'aaa/gt.png'),
        >>>     ('bbb/im.png', 'bbb/gt.png'),
        >>>     ('ccc/im.png', 'ccc/gt.png'),
        >>>     # ---
        >>>     ('ims/im-aaa.png', 'gts/gt-aaa.png'),
        >>>     ('ims/im-bbb.png', 'gts/gt-bbb.png'),
        >>>     ('ims/im-ccc.png', 'gts/gt-ccc.png'),
        >>>     # ---
        >>>     ('ims/aaa-im.png', 'gts/aaa-gt.png'),
        >>>     ('ims/bbb-im.png', 'gts/bbb-gt.png'),
        >>>     ('ims/ccc-im.png', 'gts/ccc-gt.png'),
        >>> ]
        >>> paths1, paths2 = zip(*aligned)

    """

    def comparable_unique_path_ids(paths1, paths2):
        """
        Given two unordered sets of paths (that are assumed to have some unknown
        correspondence) we find a unique id for each path in each set such that
        they can be aligned.
        """
        assert len(paths1) == len(paths2), (
            'cannot align unequal no of items {} != {}.'.format(len(paths1), len(paths2)))

        do_quick_check = True
        if do_quick_check:
            # First check the simple thing: do they have unique corresponding
            # basenames. If not we do something a bit more complex.
            simple_unique1 = list(map(_fast_basename_we, paths1))
            simple_unique_set1 = set(simple_unique1)
            if len(simple_unique_set1) == len(paths1):
                simple_unique2 = list(map(_fast_basename_we, paths2))
                simple_unique_set2 = set(simple_unique2)
                if simple_unique_set2 == simple_unique_set1:
                    return simple_unique1, simple_unique2

        safe_paths1 = _safepaths(paths1)
        safe_paths2 = _safepaths(paths2)

        # unique identifiers that should be comparable
        unique1 = shortest_unique_suffixes(safe_paths1, sep='-')
        unique2 = shortest_unique_suffixes(safe_paths2, sep='-')

        def not_comparable_msg():
            return '\n'.join([
                'paths are not comparable'
                'safe_paths1 = {}'.format(safe_paths1[0:3]),
                'safe_paths2 = {}'.format(safe_paths1[0:3]),
                'paths1 = {}'.format(safe_paths1[0:3]),
                'paths2 = {}'.format(safe_paths1[0:3]),
                'unique1 = {}'.format(unique1[0:3]),
                'unique2 = {}'.format(unique2[0:3]),
            ])

        try:
            # Assert these are unique identifiers common between paths
            assert sorted(set(unique1)) == sorted(unique1), not_comparable_msg()
            assert sorted(set(unique2)) == sorted(unique2), not_comparable_msg()
            assert sorted(unique1) == sorted(unique2), not_comparable_msg()
        except AssertionError:
            unique1 = shortest_unique_prefixes(safe_paths1, sep='-')
            unique2 = shortest_unique_prefixes(safe_paths2, sep='-')
            # Assert these are unique identifiers common between paths
            assert sorted(set(unique1)) == sorted(unique1), not_comparable_msg()
            assert sorted(set(unique2)) == sorted(unique2), not_comparable_msg()
            assert sorted(unique1) == sorted(unique2), not_comparable_msg()
        return unique1, unique2

    unique1, unique2 = comparable_unique_path_ids(paths1, paths2)

    lookup = {k: v for v, k in enumerate(unique1)}
    sortx = np.argsort([lookup[u] for u in unique2])

    sorted_paths2 = [paths2[x] for x in sortx]
    return sorted_paths2


def check_aligned(paths1, paths2):
    from os.path import basename
    if len(paths1) != len(paths2):
        return False

    # Try to short circuit common cases
    basenames1 = map(basename, paths1)
    basenames2 = map(basename, paths2)
    if all(p1 == p2 for p1, p2 in zip(basenames1, basenames2)):
        return True

    try:
        # Full case
        aligned_paths2 = align_paths(paths1, paths2)
    except AssertionError:
        return False
    return aligned_paths2 == paths2


if __name__ == '__main__':
    r"""
    CommandLine:
        xdoctest -m netharn.util.util_fname all --pygtrie
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
