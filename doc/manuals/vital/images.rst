Images
======

TODO Write an introduction to images and why they are important

Types
-----

Image
~~~~~

..  doxygenclass:: kwiver::vital::image
    :project: kwiver
    
:ref:`Doxygen continued...<image>`

Image Container
~~~~~~~~~~~~~~~

..  doxygenclass:: kwiver::vital::image_container
    :project: kwiver
    
:ref:`Doxygen continued...<image_container>`


Algorithms
----------

.. _algo_image_io:

Image I/O
~~~~~~~~~

..  doxygenclass:: kwiver::vital::algo::image_io
    :project: kwiver
    
:ref:`Doxygen continued...<image_io>`

Instantiate with: ::

   kwiver::vital::algo::image_io_sptr img_io = kwiver::vital::algo::image_io::create("<impl_name>");


============================== ====================== ========================
 Arrow & Configuration           <impl_name> value      CMake Flag to Enable  
============================== ====================== ========================
:ref:`OpenCV<ocv_image_io>`       ::create("ocv")       KWIVER_ENABLE_OPENCV  
:ref:`VXL<vxl_image_io>`          ::create("vxl")       KWIVER_ENABLE_VXL     
============================== ====================== ========================

.. _algo_image_filter:

Image Filter
~~~~~~~~~~~~

..  doxygenclass:: kwiver::vital::algo::image_filter
    :project: kwiver
    
:ref:`Doxygen continued...<image_filter>`

Instantiate with: ::

   kwiver::vital::algo::image_filter_sptr img_filter = kwiver::vital::algo::image_filter::create("<impl_name>");

============================== ====================== ========================
 Arrow & Configuration           <impl_name> value      CMake Flag to Enable  
============================== ====================== ========================
N/A                             N/A                    N/A                    
============================== ====================== ========================

** Currently there are no arrows implementing the image_filter algorithm **

.. _algo_split_image:

Split Image
~~~~~~~~~~~

..  doxygenclass:: kwiver::vital::algo::split_image
    :project: kwiver
    
:ref:`Doxygen continued...<split_image>`

Instantiate with: ::

   kwiver::vital::algo::split_image_sptr img_filter = kwiver::vital::algo::split_image::create("<impl_name>");

============================== ====================== ========================
 Arrow & Configuration           <impl_name> value      CMake Flag to Enable  
============================== ====================== ========================
:ref:`OpenCV<ocv_split_image>`    ::create("ocv")       KWIVER_ENABLE_OPENCV  
:ref:`VXL<vxl_split_image>`       ::create("vxl")       KWIVER_ENABLE_VXL     
============================== ====================== ========================

Code Example
------------

.. literalinclude:: ../../../examples/cpp/how_to_part_01_images.cpp
   :linenos:
   :language: cpp
   :lines: 30-
