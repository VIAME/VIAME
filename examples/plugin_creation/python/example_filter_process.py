#
# INSERT COPYRIGHT STATEMENT OR DELETE THIS
#

"""A pipeline process: more plumbing, but full control of the ports.

It gets an image as input, prints a configured message, and sends the image
on. The package's `__init__.py` declares it as `example_filter_process`.
"""

from viame.pipeline import process
from viame.processes.base import ViameProcess
from viame.types import ImageContainer


class ExampleFilterProcess( ViameProcess ):

    def __init__( self, conf ):
        ViameProcess.__init__( self, conf )

        # ( config name, config type, default value, description )
        self.add_config_trait( "text", "text", "Hello World",
                               "Text to display to user." )
        self.declare_config_using_trait( "text" )

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add( self.flag_required )

        # ( port name, flags )
        self.declare_input_port_using_trait( "image", required )
        self.declare_output_port_using_trait( "image", optional )

    def _configure( self ):
        self.text = self.config_value( "text" )
        self._base_configure()

    def _step( self ):
        in_img_c = self.grab_input_using_trait( "image" )

        print( "Text: " + str( self.text ) )

        self.push_to_port_using_trait( "image", ImageContainer( in_img_c.image() ) )
        self._base_step()
