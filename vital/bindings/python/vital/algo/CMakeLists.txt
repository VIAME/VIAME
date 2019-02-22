project(vital_python_algo)

include_directories(${PYBIND11_INCLUDE_DIR})

# Add trampoline
kwiver_add_python_library(py_algorithm_trampoline
  vital/algo
  trampoline/algorithm_trampoline.cxx  
)

target_link_libraries(python-vital.algo-py_algorithm_trampoline
  PUBLIC ${PYTHON_LIBRARIES}
          vital_algo
          vital_config
          vital  
)

kwiver_add_python_library(py_image_object_detector_trampoline
  vital/algo
  trampoline/image_object_detector_trampoline.cxx  
)

target_link_libraries(python-vital.algo-py_image_object_detector_trampoline
  PUBLIC ${PYTHON_LIBRARIES}
          vital_algo
          vital_config
          vital  
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

kwiver_add_python_module(${CMAKE_CURRENT_SOURCE_DIR}/simple_image_detector.py
  vital/algo
  simple_image_detector
)


kwiver_create_python_init(vital/algo
  image_object_detector
  algorithm_factory  
)

