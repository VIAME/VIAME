"""
SMQTK Exceptions - Minimal port for VIAME search functionality.
"""


class ReadOnlyError(Exception):
    """
    For when an attempt at modifying an immutable container is made.
    """


class NoUriResolutionError(Exception):
    """
    Standard exception thrown by base DataElement from_uri method when a
    subclass does not implement URI resolution.
    """


class InvalidUriError(Exception):
    """
    An invalid URI was provided.
    """

    def __init__(self, uri_value, reason):
        super(InvalidUriError, self).__init__(uri_value, reason)
        self.uri = uri_value
        self.reason = reason
