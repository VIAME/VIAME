# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process

from kwiver.vital.types import ImageContainer

class hello_world_filter( KwiverProcess ):
    """
    Example image filter process demonstrating Python plugin development.

    This example shows how to create an image filter in Python that
    integrates with the KWIVER/VIAME pipeline system. It receives images
    as input and produces processed images as output.

    The implementation is intentionally simple - it only logs a configurable
    text message and returns the input image unchanged. Use this as a template
    when creating your own image processing filters.

    Configuration:
        text (str): Message to display when processing each image.
                    Default: "Hello World"

    Input Ports:
        image: Input image to process (kwiver:image, required)

    Output Ports:
        out_image: Processed output image (kwiver:image)

    Example Pipeline Usage:
        process my_filter
          :: hello_world_filter
          :text    Applying filter...

    To Create Your Own Filter:
        1. Copy this file and rename the class
        2. Add your image processing logic in _step()
        3. Modify the image data as needed (using numpy/OpenCV)
        4. Wrap the result in ImageContainer before pushing
        5. Register your class in __init__.py
    """

    def __init__( self, conf ):
        """
        Initialize the filter process.

        Sets up configuration parameters and declares input/output ports.
        This method is called once when the pipeline is constructed.

        Args:
            conf: Process configuration object from the pipeline system
        """
        KwiverProcess.__init__( self, conf )

        # Declare configuration parameters
        # add_config_trait(key, trait_name, default_value, description)
        self.add_config_trait( "text", "text", 'Hello World',
            'Text to display to user.' )

        self.declare_config_using_trait( 'text' )

        # Define a custom output port trait for the processed image
        self.add_port_trait( 'out_image', 'image', 'Processed image' )

        # Set up port flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add( self.flag_required )

        # Declare input and output ports
        self.declare_input_port_using_trait( 'image', required )
        self.declare_output_port_using_trait( 'out_image', optional )

    def _configure( self ):
        """
        Read configuration values after pipeline initialization.

        Called after __init__ when the pipeline is configured. Use this
        to read configuration values and initialize any resources needed
        for processing.
        """
        self.text = self.config_value( 'text' )

        self._base_configure()

    def _step( self ):
        """
        Process one input image and produce the filtered output.

        This is the main processing method called for each input image.
        Implement your image processing algorithm here.

        The basic pattern is:
            1. Grab input image from port
            2. Extract the image data (numpy array)
            3. Process the image (your algorithm)
            4. Wrap result in ImageContainer and push to output port
            5. Call _base_step() to signal completion
        """
        # Grab image container from input port
        in_img_c = self.grab_input_using_trait( 'image' )

        # Get the underlying image (numpy array) from the container
        # This gives access to pixel data for processing
        in_img = in_img_c.get_image()

        # Log the configured text message
        # Replace this with your image processing algorithm
        print( "Text: " + str( self.text ) )

        # Wrap the processed image in an ImageContainer and push to output
        # In a real filter, you would modify in_img before wrapping
        self.push_to_port_using_trait( 'out_image', ImageContainer( in_img ) )

        # Signal that this step is complete
        self._base_step()
