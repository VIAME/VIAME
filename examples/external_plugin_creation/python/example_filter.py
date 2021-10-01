#
# INSERT COPYRIGHT STATEMENT OR DELETE THIS
#

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess

from vital.types import Image
from vital.types import ImageContainer

class example_filter( KwiverProcess ):
    """
    This process gets an image as input, prints out text, and sends a copy of
    the image to the output port.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.add_config_trait("text", "text", 'Hello World',
          'Text to display to user.')

        self.declare_config_using_trait('text')

        self.add_port_trait('out_image', 'image', 'Processed image')

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)
        self.declare_output_port_using_trait('out_image', optional )

    # ----------------------------------------------
    def _configure(self):
        self.text = self.config_value('text')

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')

        # Get python image from conatiner (just for show)
        in_img = in_img_c.get_image()

        # Print out text to screen
        print "Text: " + str( self.text )

        # push dummy detections object to output port
        self.push_to_port_using_trait('out_image', ImageContainer(in_img))

        self._base_step()

