# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Common utility functions for VIAME Python modules.
"""


def str2bool(v):
    """
    Convert a string representation of a boolean to an actual boolean value.

    Parameters
    ----------
    v : str or bool
        The value to convert. Accepts 'yes', 'true', 't', 'y', '1' for True
        and 'no', 'false', 'f', 'n', '0' for False (case-insensitive).

    Returns
    -------
    bool
        The boolean value.

    Raises
    ------
    ValueError
        If the string cannot be interpreted as a boolean.

    Examples
    --------
    >>> str2bool('yes')
    True
    >>> str2bool('False')
    False
    >>> str2bool(True)
    True
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError(f'Boolean value expected, got {v}')
