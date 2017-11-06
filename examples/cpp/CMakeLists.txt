
set(SOURCE "main.cpp"
           "how_to_part_01_images.cpp"
           "how_to_part_02_detections.cpp"
)

add_executable(kwiver_examples ${SOURCE})
target_link_libraries(kwiver_examples 
                      vital_algo 
                      kwiver_algo_ocv kwiver_algo_vxl)

