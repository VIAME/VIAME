Detector Data Types and Related Algorithms
==========================================

Detected Object Set
-------------------

.. _vital_detected_object_set:
..  doxygenclass:: kwiver::vital::detected_object_set
    :project: kwiver
    :members:

Detected Object
---------------

.. _vital_detected_object:
..  doxygenclass:: kwiver::vital::detected_object
    :project: kwiver
    :members:
    
Detected Object Type
--------------------

.. _vital_detected_object_type:
..  doxygenclass:: kwiver::vital::detected_object_type
    :project: kwiver
    :members:
    
Bounding Box
------------

.. _vital_bounding_box:
..  doxygenclass:: kwiver::vital::bounding_box
    :project: kwiver
    :members:
    
Descriptor
----------

.. _vital_descriptor:
..  doxygenclass:: kwiver::vital::descriptor
    :project: kwiver
    :members:

Image Object Detector Algorithm
-------------------------------
.. _vital_image_object_detector:

Instantiate with: ::

   kwiver::vital::algo::image_object_detector_sptr detector = kwiver::vital::algo::image_object_detector::create("<impl_name>");

======================================= ==================== ========================
 Arrow & Configuration                   <impl_name> options    CMake Flag to Enable 
======================================= ==================== ========================
:ref:`Darknet<darknet_detector>`        darknet                KWIVER_ENABLE_DARKNET 
:ref:`Hough<ocv_hough_circle_detector>` hough                  KWIVER_ENABLE_OPENCV  
======================================= ==================== ========================

..  doxygenclass:: kwiver::vital::algo::image_object_detector
    :project: kwiver
    :members:

    
Train Detector Algorithm
------------------------
.. _vital_train_detector:

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

Detected Object Filter Algorithm
--------------------------------
.. _vital_detected_object_filter:

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

    
Draw Detected Object Set Algorithm
----------------------------------
.. _vital_draw_detected_object_set:

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

Detected Object Set Input Algorithm
------------------------------------
.. _vital_detected_object_set_input:

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
    
Detected Object Set Output Algorithm
------------------------------------
.. _vital_detected_object_set_output:

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