set(CMAKE_FOLDER "Examples")
set(EXAMPLE_DIR ${CMAKE_BINARY_DIR}/examples)
file(MAKE_DIRECTORY "${EXAMPLE_DIR}/pipelines/output")
file(MAKE_DIRECTORY "${EXAMPLE_DIR}/models")

add_subdirectory(cpp)
add_subdirectory(external)

# Configure all the local example pipe files
configure_file(pipelines/burnout_on_video.pipe.in ${EXAMPLE_DIR}/pipelines/burnout_on_video.pipe @ONLY )
configure_file(pipelines/image_list.txt.in ${EXAMPLE_DIR}/pipelines/image_list.txt @ONLY )
configure_file(pipelines/image_display.pipe.in ${EXAMPLE_DIR}/pipelines/image_display.pipe @ONLY )
configure_file(pipelines/image_serialize.pipe.in ${EXAMPLE_DIR}/pipelines/image_serialize.pipe @ONLY )
configure_file(pipelines/images_to_python.pipe.in ${EXAMPLE_DIR}/pipelines/images_to_python.pipe @ONLY )
configure_file(pipelines/hough_detector.pipe.in ${EXAMPLE_DIR}/pipelines/hough_detector.pipe @ONLY )
configure_file(pipelines/hough_detector_images.txt.in ${EXAMPLE_DIR}/pipelines/hough_detector_images.txt @ONLY )
configure_file(pipelines/number_flow.pipe.in ${EXAMPLE_DIR}/pipelines/number_flow.pipe @ONLY )
configure_file(pipelines/number_flow_python.pipe.in ${EXAMPLE_DIR}/pipelines/number_flow_python.pipe @ONLY )
configure_file(pipelines/video_display.pipe.in ${EXAMPLE_DIR}/pipelines/video_display.pipe @ONLY )
configure_file(pipelines/example_detector_on_image.pipe.in ${EXAMPLE_DIR}/pipelines/example_detector_on_image.pipe @ONLY )
configure_file(pipelines/example_pydetector_on_image.pipe.in ${EXAMPLE_DIR}/pipelines/example_pydetector_on_image.pipe @ONLY )
configure_file(pipelines/example_detector_on_video.pipe.in ${EXAMPLE_DIR}/pipelines/example_detector_on_video.pipe @ONLY )

configure_file(pipelines/descriptor.pipe.in ${EXAMPLE_DIR}/pipelines/descriptor.pipe @ONLY )
configure_file(pipelines/SMQTK.pipe.in ${EXAMPLE_DIR}/pipelines/SMQTK.pipe @ONLY )

configure_file(pipelines/images_to_kwa.pipe.in ${EXAMPLE_DIR}/pipelines/images_to_kwa.pipe @ONLY )
configure_file(pipelines/images_to_kwa_split.pipe.in ${EXAMPLE_DIR}/pipelines/images_to_kwa_split.pipe @ONLY )
configure_file(pipelines/stabilize_images.pipe.in ${EXAMPLE_DIR}/pipelines/stabilize_images.pipe @ONLY )
configure_file(pipelines/track_features.pipe.in ${EXAMPLE_DIR}/pipelines/track_features.pipe @ONLY )

configure_file(models/default_burnout_600_iters ${EXAMPLE_DIR}/models/default_burnout_600_iters COPYONLY )
