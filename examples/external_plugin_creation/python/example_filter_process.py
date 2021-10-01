#
# INSERT COPYRIGHT STATEMENT OR DELETE THIS
#

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess

from vital.types import Image
from vital.types import ImageContainer

class example_filter_process( KwiverProcess ):
    """
    This process gets an image as input, prints out text, and sends a copy of
    the image to the output port.
    """
    # --------------------------------------------------------------------------
    def __init__( self, conf ):
        KwiverProcess.__init__( self, conf )

        self.add_config_trait( "text",    # Config name
          "text",                         # Config type
          'Hello World',                  # Default value
          'Text to display to user.' )    # Config

        self.declare_config_using_trait( 'text' )  # Register config name

        # Would be used if declaring a new output field 'out_image' instead
        # of the default 'image' output port name defined in KwiverProcess
        #self.add_port_trait( 'out_image', 'image', 'Processed image' )

        # set up required port flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add( self.flag_required )

        # declare our input and output ports ( port-name, flags )
        self.declare_input_port_using_trait( 'image', required )
        self.declare_output_port_using_trait( 'image', optional )

    # --------------------------------------------------------------------------
    def _configure( self ):
        self.text = self.config_value( 'text' ) # Read config file from file
        self._base_configure()

    # --------------------------------------------------------------------------
    def _step( self ):
        # Get c image pointer from incoming pipeline
        in_img_c = self.grab_input_using_trait( 'image' )

        # Get python image from conatiner (not used here, just for show)
        in_img = in_img_c.image()

        # Print out text to screen, just so we're doing something here
        print( "Text: " + str( self.text ) )

        # push dummy detections object to output port
        self.push_to_port_using_trait( 'image', ImageContainer( in_img ) )

        self._base_step()

