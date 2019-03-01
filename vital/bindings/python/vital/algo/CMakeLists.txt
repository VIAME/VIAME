project(vital_python_algo)

include_directories(${PYBIND11_INCLUDE_DIR})

# Add trampolines

set( vital_algo_trampoline
    trampoline/algorithm_trampoline.tcc
    trampoline/image_object_detector_trampoline.tcc
)

kwiver_add_library( 
  vital_algo_trampoline
  ${vital_algo_trampoline}
)

set_target_properties(vital_algo_trampoline PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(vital_algo_trampoline
  PUBLIC ${PYTHON_LIBRARIES}
          vital_algo
          vital_config
          vital
          vital_python_util  
)


kwiver_add_python_library(image_object_detector
  vital/algo
  image_object_detector.cxx 
)

target_link_libraries(python-vital.algo-image_object_detector
  PUBLIC ${PYTHON_LIBRARIES}
          vital
          vital_config
          vital_algo
)

kwiver_add_python_library(algorithm_factory
  vital/algo
  algorithm_factory.cxx  
)

target_link_libraries(python-vital.algo-algorithm_factory
  PUBLIC ${PYTHON_LIBRARIES}
          vital
          vital_algo
)


kwiver_create_python_init(vital/algo
  image_object_detector
  algorithm_factory  
)

add_subdirectory(py)
