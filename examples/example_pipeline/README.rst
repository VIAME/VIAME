
============================
New Module Creation Examples
============================

This document corresponds to `this runable example`_ of `these example simple plugins`_,
alongside `these example plugin templates`_. Additionally, all of the former can be found in
[viame-install]/examples/example_pipeline folder,
[viame-source]/plugins/hello_world folder,
and [viame-source]/plugins/templates folder in a VIAME installation, respectively.
Throughout these folders are example object detectors, image filters, and image classifier
implementations written in both Python and C++.

.. _this runable example: https://github.com/VIAME/VIAME/tree/master/examples/example_pipeline
.. _these example simple plugins: https://github.com/VIAME/VIAME/tree/master/plugins/hello_world
.. _these example plugin templates: https://github.com/VIAME/VIAME/tree/master/plugins/templates


**********************************
Simple C++ Detector Plugin Example
**********************************

A new detector plugin can be added by creating a class that implements the
kwiver::vital::algo::image_object_detector interface. This interface
is defined in an abstract base class in file vital/algo/image_object_detector.h.
Similar interfaces exist for several other types of functions.

The directory `plugins/templates/cxx` contains files that can be used
as a starting point for implementing new detectors. These files
contain markers in the form `@text@` that are to be replaced with the
string (the title) of your new detector.

All files in that directory should be copied to a new directory and
renamed as appropriate. The files template_detector.{cxx,h} should be
renamed to a name that indicates the specific detector being
implemented.

The CMakeLists.txt and register_algorithms.cxx files should keep their
original names. Change the following place holders in all files to
personalize the detector.

@template@ - name of the detector.

@template_lib@ - name of the plugin library that will contain the
detector. Can be the same name as the detector.

@template_dir@ - name of the source subdirectory containing the detector
files. For example if the detector is in the directory plugins/ex_fish_detector,
then 'template_dir' should be replaced with 'ex_fish_detector'.

The place holders also appear in capital letters indicating that the
replacement string should be capitalized.

The main work that has to be done to integrate a detector into the
VIAME framework is to convert the input image from the VIAME format to
the format needed by the detector, and to convert the detections to a
detected_object_set as needed by the framework.

Many detectors take images in OpenCV matrix format. This data structure
can be extracted from the image_container_sptr that is available using
the following code:

::

    // input image is kwiver::vital::image_container_sptr image_data
    // CV format image is extracted using the following line
    cv::Mat cv_image = kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image() );

Now that you have the image in a compatible format, it can be passed
to the detector. Detectors usually return a set of bounding boxes,
each annotated with one or more classification labels. These boxes can
be converted to a detected_object_set using the following pseudo-code.

::

    // Allocate a detected object set that we will fill with new detections
    auto detected_objects = std::make_shared<kwiver::vital::detected_object_set>();

    FOREACH bounding-box returned from detector

        // Create a bounding box from the values returned. The new box takes
        // coordinates in the following order: left, top, right, bottom.
        // If the detector does not return exactly these values, they are
        // easy to calculate
        kwiver::vital::bounding_box_d bbox( left, top, right, bot);

        // Create a new detected object type structure. This is used to hold the
        // classification labels and associated probabilities or scores.
        auto dot = std::make_shared< kwiver::vital::detected_object_type >();

        FOREACH pair of classification label and score
            // Add the class name and probability to the detected object type
            dot->set_score( class_name, probability );
        END_FOREACH

        // Now that we have processed one detected object (as defined by a bounding box)
        // it has to be added to the detected_object_set
        detected_objects->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ));
    END_FOREACH

    // When all detections have been processed, the detected object set for this input
    // image is just returned from the detect() method
    return detected_objects;

**********************
Python Detector Plugin
**********************

Similarly to the above C++ object detector, the python templates in the above directory
can be copied into a new plugin module, and the template keywords replaced with a module
name of your choosing.
