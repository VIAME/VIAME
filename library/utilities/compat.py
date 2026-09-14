"""Small compatibility helpers for the python plugins.

Kept deliberately dependency-free so that every plugin can import it.
"""


def strtobool( value ):
    """Convert a string representation of truth to 1 or 0.

    Replaces distutils.util.strtobool, which was removed along with the rest
    of distutils in python 3.12. True values are y, yes, t, true, on and 1;
    false values are n, no, f, false, off and 0. Anything else is a ValueError.
    """
    value = str( value ).lower()

    if value in ( "y", "yes", "t", "true", "on", "1" ):
        return 1
    if value in ( "n", "no", "f", "false", "off", "0" ):
        return 0

    raise ValueError( "invalid truth value {!r}".format( value ) )
