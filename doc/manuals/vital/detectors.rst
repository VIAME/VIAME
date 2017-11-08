Detector Data Types and Related Algorithms
==========================================

.. _vital_detected_object_set:

Detected Object Set
-------------------

..  doxygenclass:: kwiver::vital::detected_object_set
    :project: kwiver
    :members:

.. _vital_detected_object:

Detected Object
---------------

..  doxygenclass:: kwiver::vital::detected_object
    :project: kwiver
    :members:
    
.. _vital_detected_object_type:

Detected Object Type
--------------------

..  doxygenclass:: kwiver::vital::detected_object_type
    :project: kwiver
    :members:
    
.. _vital_bounding_box:

Bounding Box
------------

..  doxygenclass:: kwiver::vital::bounding_box
    :project: kwiver
    :members:
    
.. _vital_descriptor:

Descriptor
----------

..  doxygenclass:: kwiver::vital::descriptor
    :project: kwiver
    :members:

.. _vital_image_object_detector:

Image Object Detector Algorithm
-------------------------------

Instantiate with: ::

   kwiver::vital::algo::image_object_detector_sptr detector = kwiver::vital::algo::image_object_detector::create("<impl_name>");

======================================= ==================== ========================
 Arrow & Configuration                   <impl_name> options    CMake Flag to Enable 
======================================= ==================== ========================
:ref:`Example<core_example_detector>`   example_detector       KWIVER_ENABLE_ARROWS  
:ref:`Hough<ocv_hough_circle_detector>` hough_circle           KWIVER_ENABLE_OPENCV  
:ref:`Darknet<darknet_detector>`        darknet                KWIVER_ENABLE_DARKNET 
======================================= ==================== ========================

..  doxygenclass:: kwiver::vital::algo::image_object_detector
    :project: kwiver
    :members:

.. _vital_train_detector:

Train Detector Algorithm
------------------------

Instantiate with: ::

   kwiver::vital::algo::train_detector_sptr trainer = kwiver::vital::algo::train_detector::create("<impl_name>");

=============================== ====================== ========================
 Arrow & Configuration           <impl_name> options    CMake Flag to Enable   
=============================== ====================== ========================
:ref:`Darknet<darknet_trainer>` darknet                 KWIVER_ENABLE_DARKNET  
=============================== ====================== ========================

..  doxygenclass:: kwiver::vital::algo::train_detector
    :project: kwiver
    :members:

.. _vital_detected_object_filter:

Detected Object Filter Algorithm
--------------------------------

Instantiate with: ::

   kwiver::vital::algo::detected_object_filter_sptr filter = kwiver::vital::algo::detected_object_filter::create("<impl_name>");

========================================= ======================= ========================
 Arrow & Configuration                      <impl_name> options    CMake Flag to Enable   
========================================= ======================= ========================
:ref:`Core<core_class_probablity_filter>` class_probablity_filter  KWIVER_ENABLE_ARROWS   
========================================= ======================= ========================

..  doxygenclass:: kwiver::vital::algo::detected_object_filter
    :project: kwiver
    :members:

.. _vital_draw_detected_object_set:

Draw Detected Object Set Algorithm
----------------------------------

Instantiate with: ::

   kwiver::vital::algo::draw_detected_object_set_sptr draw = kwiver::vital::algo::draw_detected_object_set::create("<impl_name>");

=========================================== ====================== ========================
 Arrow & Configuration                        <impl_name> options    CMake Flag to Enable  
=========================================== ====================== ========================
:ref:`OpenCV<ocv_draw_detected_object_set>`          ocv             KWIVER_ENABLE_OPENCV  
=========================================== ====================== ========================

..  doxygenclass:: kwiver::vital::algo::draw_detected_object_set
    :project: kwiver
    :members:

.. _vital_detected_object_set_input:

Detected Object Set Input Algorithm
------------------------------------

Instantiate with: ::

   kwiver::vital::algo::detected_object_set_input_sptr detec_in = kwiver::vital::algo::detected_object_set_input::create("<impl_name>");

================================================ ====================== ========================
 Arrow & Configuration                            <impl_name> options    CMake Flag to Enable  
================================================ ====================== ========================
:ref:`CSV<core_detected_object_set_input_csv>`          csv              KWIVER_ENABLE_ARROWS  
:ref:`KW18<core_detected_object_set_input_kw18>`        kw18             KWIVER_ENABLE_ARROWS  
================================================ ====================== ========================

..  doxygenclass:: kwiver::vital::algo::detected_object_set_input
    :project: kwiver
    :members:
    
.. _vital_detected_object_set_output:

Detected Object Set Output Algorithm
------------------------------------

Instantiate with: ::

   kwiver::vital::algo::detected_object_set_output_sptr detec_out = kwiver::vital::algo::detected_object_set_output::create("<impl_name>");

================================================= ====================== ========================
 Arrow & Configuration                             <impl_name> options    CMake Flag to Enable  
================================================= ====================== ========================
:ref:`CSV<core_detected_object_set_output_csv>`          csv              KWIVER_ENABLE_ARROWS  
:ref:`KW18<core_detected_object_set_output_kw18>`        kw18             KWIVER_ENABLE_ARROWS  
================================================= ====================== ========================

..  doxygenclass:: kwiver::vital::algo::detected_object_set_output
    :project: kwiver
    :members:
    
Code Example
------------

.. literalinclude:: ../../../examples/cpp/how_to_part_02_detections.cpp
   :language: cpp
   :lines: 30-