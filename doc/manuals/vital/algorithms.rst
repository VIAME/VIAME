Vital Algorithms
================

Vital contains a collection of pure virtual algorithm interfaces to extend with specific methodology.
There is an algorithm :ref:`abstract base class<algorithm>` for all algorithm implementations.
Next there is an intermediate :ref:`algorithm template<algorithm_def>` that is used to provide abstract algorithm definitions for specific functionality.


Provided Algorithm Base Classes
-------------------------------

Vital provides a collection of algorithm base classes that represent specific functionality and are instantiated as KWIVER arrows. 
KWIVER end-users may use a provided algorithm's arrow implementation, create their own implementation, 
and even create a new algorithm template to define new algorithm functionality previously undefined in kwiver.

===================================== ======================================================
:ref:`analyze_tracks<analyze_tracks>` .. doxygenclass:: kwiver::vital::algo::analyze_tracks
                                         :no-link:
===================================== ======================================================

================================================================ =========================================================== =================================================================== =================================================================
:ref:`analyze_tracks<analyze_tracks>`                            :ref:`bundle_adjust<bundle_adjust>`                         :ref:`close_loops<close_loops>`                                     :ref:`compute_ref_homography<compute_ref_homography>`            
:ref:`compute_stereo_depth_map<compute_stereo_depth_map>`        :ref:`compute_track_descriptors<compute_track_descriptors>` :ref:`convert_image<convert_image>`                                 :ref:`detect_features<detect_features>`                          
:ref:`detected_object_filter<detected_object_filter>`            :ref:`detected_object_set_input<detected_object_set_input>` :ref:`detected_object_set_output<detected_object_set_output>`       :ref:`draw_detected_object_set<draw_detected_object_set>`        
:ref:`draw_tracks<draw_tracks>`                                  :ref:`dynamic_configuration<dynamic_configuration>`         :ref:`estimate_canonical_transform<estimate_canonical_transform>`   :ref:`estimate_essential_matrix<estimate_essential_matrix>`      
:ref:`estimate_fundamental_matrix<estimate_fundamental_matrix>`  :ref:`estimate_homography<estimate_homography>`             :ref:`estimate_similarity_transform<estimate_similarity_transform>` :ref:`extract_descriptors<extract_descriptors>`                  
:ref:`feature_descriptor_io<feature_descriptor_io>`              :ref:`filter_features<filter_features>`                     :ref:`filter_tracks<filter_tracks>`                                 :ref:`formulate_query<formulate_query>`                          
:ref:`image_filter<image_filter>`                                :ref:`image_io<image_io>`                                   :ref:`image_object_detector<image_object_detector>`                 :ref:`initialize_cameras_landmarks<initialize_cameras_landmarks>`
:ref:`match_features<match_features>`                            :ref:`optimize_cameras<optimize_cameras>`                   :ref:`refine_detections<refine_detections>`                         :ref:`track_descriptor_set_input<track_descriptor_set_input>`    
:ref:`track_descriptor_set_output<track_descriptor_set_output>`  :ref:`track_features<track_features>`                       :ref:`train_detector<train_detector>`                               :ref:`triangulate_landmarks<triangulate_landmarks>`              
================================================================ =========================================================== =================================================================== =================================================================
