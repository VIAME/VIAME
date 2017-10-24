Vital Doxygen
=============

Types
-----

Other
-----

There are various other vital types that are also used to help direct algorithms or hold specific data associated with an image.

============================================= =========================================== ===========================================
:ref:`camera<camera>`                         :ref:`camera_intrinsics<camera_intrinsics>`                                            
:ref:`rgb_color<rgb_color>`                   :ref:`covariance<covariance>`               :ref:`descriptor<descriptor>`              
:ref:`descriptor_request<descriptor_request>` :ref:`descriptor_set<descriptor_set>`                                                  
============================================= =========================================== ===========================================


Image
~~~~~

.. _image:
..  doxygenclass:: kwiver::vital::image
    :project: kwiver
    :members:
    
.. _image_container:
..  doxygenclass:: kwiver::vital::image_container
    :project: kwiver
    :members:

Detections
~~~~~~~~~~

.. _bounding_box:
..  doxygenclass:: kwiver::vital::bounding_box
    :project: kwiver
    :members:
    

.. _detected_object:
..  doxygenclass:: kwiver::vital::detected_object
    :project: kwiver
    :members:
    
.. _detected_object_set:
..  doxygenclass:: kwiver::vital::detected_object_set
    :project: kwiver
    :members:
    

Other
~~~~~


.. _camera:
..  doxygenclass:: kwiver::vital::camera
    :project: kwiver
    :members:

.. _camera_intrinsics:
..  doxygenclass:: kwiver::vital::camera_intrinsics
    :project: kwiver
    :members:
    
.. _rgb_color:
..  doxygenstruct:: kwiver::vital::rgb_color
    :project: kwiver
    :members:

.. _covariance:
..  doxygenclass:: kwiver::vital::covariance_
    :project: kwiver
    :members:
    
.. _descriptor:
..  doxygenclass:: kwiver::vital::descriptor
    :project: kwiver
    :members:

.. _descriptor_request:
..  doxygenclass:: kwiver::vital::descriptor_request
    :project: kwiver
    :members:
    
.. _descriptor_set:
..  doxygenclass:: kwiver::vital::descriptor_set
    :project: kwiver
    :members:

Algorithms
----------

Base Types
~~~~~~~~~~

.. _algorithm:
..  doxygenclass:: kwiver::vital::algorithm
    :project: kwiver
    :members:
    
.. _algorithm_def:
..  doxygenclass:: kwiver::vital::algorithm_def
    :project: kwiver
    :members:

Functionality
~~~~~~~~~~~~~


.. _analyze_tracks:
..  doxygenclass:: kwiver::vital::algo::analyze_tracks
    :project: kwiver
    :members:


.. _bundle_adjust:
..  doxygenclass:: kwiver::vital::algo::bundle_adjust
    :project: kwiver
    :members:


.. _close_loops:
..  doxygenclass:: kwiver::vital::algo::close_loops
    :project: kwiver
    :members:


.. _compute_ref_homography:
..  doxygenclass:: kwiver::vital::algo::compute_ref_homography
    :project: kwiver
    :members:


.. _compute_stereo_depth_map:
..  doxygenclass:: kwiver::vital::algo::compute_stereo_depth_map
    :project: kwiver
    :members:


.. _compute_track_descriptors:
..  doxygenclass:: kwiver::vital::algo::compute_track_descriptors
    :project: kwiver
    :members:


.. _convert_image:
..  doxygenclass:: kwiver::vital::algo::convert_image
    :project: kwiver
    :members:


.. _detect_features:
..  doxygenclass:: kwiver::vital::algo::detect_features
    :project: kwiver
    :members:


.. _detected_object_filter:
..  doxygenclass:: kwiver::vital::algo::detected_object_filter
    :project: kwiver
    :members:


.. _detected_object_set_input:
..  doxygenclass:: kwiver::vital::algo::detected_object_set_input
    :project: kwiver
    :members:


.. _detected_object_set_output:
..  doxygenclass:: kwiver::vital::algo::detected_object_set_output
    :project: kwiver
    :members:


.. _draw_detected_object_set:
..  doxygenclass:: kwiver::vital::algo::draw_detected_object_set
    :project: kwiver
    :members:


.. _draw_tracks:
..  doxygenclass:: kwiver::vital::algo::draw_tracks
    :project: kwiver
    :members:


.. _dynamic_configuration:
..  doxygenclass:: kwiver::vital::algo::dynamic_configuration
    :project: kwiver
    :members:


.. _estimate_canonical_transform:
..  doxygenclass:: kwiver::vital::algo::estimate_canonical_transform
    :project: kwiver
    :members:


.. _estimate_essential_matrix:
..  doxygenclass:: kwiver::vital::algo::estimate_essential_matrix
    :project: kwiver
    :members:


.. _estimate_fundamental_matrix:
..  doxygenclass:: kwiver::vital::algo::estimate_fundamental_matrix
    :project: kwiver
    :members:


.. _estimate_homography:
..  doxygenclass:: kwiver::vital::algo::estimate_homography
    :project: kwiver
    :members:


.. _estimate_similarity_transform:
..  doxygenclass:: kwiver::vital::algo::estimate_similarity_transform
    :project: kwiver
    :members:

.. _extract_descriptors:
..  doxygenclass:: kwiver::vital::algo::extract_descriptors
    :project: kwiver
    :members:


.. _feature_descriptor_io:
..  doxygenclass:: kwiver::vital::algo::feature_descriptor_io
    :project: kwiver
    :members:


.. _filter_features:
..  doxygenclass:: kwiver::vital::algo::filter_features
    :project: kwiver
    :members:


.. _filter_tracks:
..  doxygenclass:: kwiver::vital::algo::filter_tracks
    :project: kwiver
    :members:


.. _formulate_query:
..  doxygenclass:: kwiver::vital::algo::formulate_query
    :project: kwiver
    :members:


.. _image_filter:
..  doxygenclass:: kwiver::vital::algo::image_filter
    :project: kwiver
    :members:


.. _image_io:
..  doxygenclass:: kwiver::vital::algo::image_io
    :project: kwiver
    :members:


.. _image_object_detector:
..  doxygenclass:: kwiver::vital::algo::image_object_detector
    :project: kwiver
    :members:


.. _initialize_cameras_landmarks:
..  doxygenclass:: kwiver::vital::algo::initialize_cameras_landmarks
    :project: kwiver
    :members:


.. _match_features:
..  doxygenclass:: kwiver::vital::algo::match_features
    :project: kwiver
    :members:


.. _optimize_cameras:
..  doxygenclass:: kwiver::vital::algo::optimize_cameras
    :project: kwiver
    :members:


.. _refine_detections:
..  doxygenclass:: kwiver::vital::algo::refine_detections
    :project: kwiver
    :members:

.. _split_image:
..  doxygenclass:: kwiver::vital::algo::split_image
    :project: kwiver
    :members:

.. _track_descriptor_set_input:
..  doxygenclass:: kwiver::vital::algo::track_descriptor_set_input
    :project: kwiver
    :members:


.. _track_descriptor_set_output:
..  doxygenclass:: kwiver::vital::algo::track_descriptor_set_output
    :project: kwiver
    :members:


.. _track_features:
..  doxygenclass:: kwiver::vital::algo::track_features
    :project: kwiver
    :members:


.. _train_detector:
..  doxygenclass:: kwiver::vital::algo::train_detector
    :project: kwiver
    :members:


.. _triangulate_landmarks:
..  doxygenclass:: kwiver::vital::algo::triangulate_landmarks
    :project: kwiver
    :members:


.. _uuid_factory:
..  doxygenclass:: kwiver::vital::algo::uuid_factory
    :project: kwiver
    :members:


.. _video_input:
..  doxygenclass:: kwiver::vital::algo::video_input
    :project: kwiver
    :members:

