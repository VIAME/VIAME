#
# INSERT COPYRIGHT STATEMENT OR DELETE THIS
#

"""An image filter algorithm: the simpler of the two ways to add python.

It implements an interface -- `ImageFilter` here -- and a pipeline selects it
by name wherever that interface is configured, for example
`:filter:type example_filter` in an `image_filter` process. The package's
`__init__.py` declares it, so nothing is imported until it is used.
"""

from viame.algo import ImageFilter
from viame.types import Image, ImageContainer


class ExampleFilter( ImageFilter ):
    """Prints a configured message and passes the image through."""

    def __init__( self ):
        ImageFilter.__init__( self )
        self._text = "Hello World"

    def get_configuration( self ):
        cfg = super( ImageFilter, self ).get_configuration()
        cfg.set_value( "text", self._text )
        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )
        self._text = str( cfg.get_value( "text" ) )

    def check_configuration( self, cfg ):
        return True

    def filter( self, image_container ):
        print( "Text: " + self._text )
        return ImageContainer( Image( image_container.image().asarray() ) )
