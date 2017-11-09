Image Data Types and Related Algorithms
=======================================

.. _vital_image:

Image
----------

..  doxygenclass:: kwiver::vital::image
    :project: kwiver
    :members:

.. _vital_timestamp:

Time Stamp
----------

..  doxygenclass:: kwiver::vital::timestamp
    :project: kwiver
    :members:

    
.. _vital_image_container:

Image Container
--------------------

..  doxygenclass:: kwiver::vital::image_container
    :project: kwiver
    :members:


.. _vital_image_io:

Image I/O Algorithm
-------------------

Instantiate with: ::

   kwiver::vital::algo::image_io_sptr img_io = kwiver::vital::algo::image_io::create("<impl_name>");

============================== ====================== ========================
 Arrow & Configuration           <impl_name> options    CMake Flag to Enable  
============================== ====================== ========================
:ref:`OpenCV<ocv_image_io>`             ocv             KWIVER_ENABLE_OPENCV  
:ref:`VXL<vxl_image_io>`                vxl             KWIVER_ENABLE_VXL     
============================== ====================== ========================

..  doxygenclass:: kwiver::vital::algo::image_io
    :project: kwiver
    :members:

.. _vital_convert_image:

Convert Image Algorithm
-----------------------

Instantiate with: ::

   kwiver::vital::algo::convert_image_sptr img_bypas = kwiver::vital::algo::convert_image::create("<impl_name>");

====================================== ====================== ========================
 Arrow & Configuration                  <impl_name> options    CMake Flag to Enable  
====================================== ====================== ========================
:ref:`Core<core_convert_image_bypass>`  bypass                 KWIVER_ENABLE_ARROWS 
====================================== ====================== ========================

..  doxygenclass:: kwiver::vital::algo::convert_image
    :project: kwiver
    :members:


.. _vital_image_filter:

Image Filter Algorithm
----------------------

Instantiate with: ::

   kwiver::vital::algo::image_filter_sptr img_filter = kwiver::vital::algo::image_filter::create("<impl_name>");

============================== ====================== ========================
 Arrow & Configuration           <impl_name> options    CMake Flag to Enable  
============================== ====================== ========================
N/A                             N/A                    N/A                    
============================== ====================== ========================

** Currently there are no arrows implementing the image_filter algorithm **

..  doxygenclass:: kwiver::vital::algo::image_filter
    :project: kwiver
    :members:

    
.. _vital_split_image:

Split Image Algorithm
---------------------

Instantiate with: ::

   kwiver::vital::algo::split_image_sptr img_split = kwiver::vital::algo::split_image::create("<impl_name>");

============================== ===================== ========================
 Arrow & Configuration          <impl_name> options    CMake Flag to Enable  
============================== ===================== ========================
:ref:`OpenCV<ocv_split_image>`          ocv            KWIVER_ENABLE_OPENCV  
:ref:`VXL<vxl_split_image>`             vxl            KWIVER_ENABLE_VXL     
============================== ===================== ========================

..  doxygenclass:: kwiver::vital::algo::split_image
    :project: kwiver
    :members:
    

.. _vital_video_input:

Video Input Algorithm
-----------------------

Instantiate with: ::

   kwiver::vital::algo::video_input_sptr img_bypas = kwiver::vital::algo::video_input::create("<impl_name>");

========================================= ====================== ========================
 Arrow & Configuration                     <impl_name> options    CMake Flag to Enable  
========================================= ====================== ========================
:ref:`VXL<vxl_vidl_ffmpeg_video_input>`        vidl_ffmpeg        KWIVER_ENABLE_VXL     
========================================= ====================== ========================

..  doxygenclass:: kwiver::vital::algo::convert_image
    :project: kwiver
    :members:

Code Example
------------

.. literalinclude:: ../../../examples/cpp/how_to_part_01_images.cpp
   :language: cpp
   :lines: 30-
