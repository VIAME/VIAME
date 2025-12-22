# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process

from kwiver.vital.types import DetectedObjectSet

class hello_world_detector( KwiverProcess ):
    """
    Example object detector process demonstrating Python plugin development.

    This example shows how to create an object detector in Python that
    integrates with the KWIVER/VIAME pipeline system. It receives images
    as input and produces detection sets as output.

    The implementation is intentionally simple - it only logs a configurable
    text message and returns an empty detection set. Use this as a template
    when creating your own detection algorithms.

    Configuration:
        text (str): Message to display when processing each image.
                    Default: "Hello World"

    Input Ports:
        image: Input image to process (kwiver:image, required)

    Output Ports:
        detected_object_set: Detection results (kwiver:detected_object_set)

    Example Pipeline Usage:
        process my_detector
          :: hello_world_detector
          :text    Processing frame...

    To Create Your Own Detector:
        1. Copy this file and rename the class
        2. Add your detection logic in _step()
        3. Create DetectedObject instances and add to DetectedObjectSet
        4. Register your class in __init__.py
    """

    def __init__( self, conf ):
        """
        Initialize the detector process.

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

        # Set up port flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add( self.flag_required )

        # Declare input and output ports using standard traits
        # Traits define the data type that flows through the port
        self.declare_input_port_using_trait( 'image', required )
        self.declare_output_port_using_trait( 'detected_object_set', optional )

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
        Process one input image and produce detections.

        This is the main processing method called for each input image.
        Implement your detection algorithm here.

        The basic pattern is:
            1. Grab input data from ports
            2. Process the data (your algorithm)
            3. Push results to output ports
            4. Call _base_step() to signal completion
        """
        # Grab image container from input port
        in_img_c = self.grab_input_using_trait( 'image' )

        # Get the underlying image (numpy array) from the container
        # This gives access to pixel data for processing
        in_img = in_img_c.image()

        # Log the configured text message
        # Replace this with your detection algorithm
        print( "Text: " + str( self.text ) )

        # Create and push detection results
        # In a real detector, you would add DetectedObject instances here
        detections = DetectedObjectSet()
        self.push_to_port_using_trait( 'detected_object_set', detections )

        # Signal that this step is complete
        self._base_step()
