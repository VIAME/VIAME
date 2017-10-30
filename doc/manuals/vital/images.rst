Images
======

Image Type
----------

.. _image:
..  doxygenclass:: kwiver::vital::image
    :project: kwiver
    :members:
    
Time Stamp
----------

.. _timestamp:
..  doxygenclass:: kwiver::vital::timestamp
    :project: kwiver
    :members:

Image Container Type
--------------------

.. _image_container:
..  doxygenclass:: kwiver::vital::image_container
    :project: kwiver
    :members:


.. _algo_image_io:

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

.. _image_io:
..  doxygenclass:: kwiver::vital::algo::image_io
    :project: kwiver
    :members:

.. _algo_image_filter:

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

.. _image_filter:
..  doxygenclass:: kwiver::vital::algo::image_filter
    :project: kwiver
    :members:

.. _algo_split_image:

Split Image Algorithm
---------------------

Instantiate with: ::

   kwiver::vital::algo::split_image_sptr img_filter = kwiver::vital::algo::split_image::create("<impl_name>");

============================== ===================== ========================
 Arrow & Configuration          <impl_name> options    CMake Flag to Enable  
============================== ===================== ========================
:ref:`OpenCV<ocv_split_image>`          ocv            KWIVER_ENABLE_OPENCV  
:ref:`VXL<vxl_split_image>`             vxl            KWIVER_ENABLE_VXL     
============================== ===================== ========================

.. _split_image:
..  doxygenclass:: kwiver::vital::algo::split_image
    :project: kwiver
    :members:

Code Example
------------

.. literalinclude:: ../../../examples/cpp/how_to_part_01_images.cpp
   :linenos:
   :language: cpp
   :lines: 30-
