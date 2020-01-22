project(vital_python_algo)

include_directories(${pybind11_INCLUDE_DIR})

kwiver_add_python_library(algorithm
  vital/algo
  algorithm.h
  image_object_detector.h
  algorithm.cxx
  image_object_detector.cxx
  algorithm_module.cxx
  trampoline/algorithm_trampoline.txx
  trampoline/image_object_detector_trampoline.txx
)


target_link_libraries(python-vital.algo-algorithm
  PUBLIC ${PYTHON_LIBRARIES}
          vital
          vital_config
          vital_algo
          vital_python_util
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
  algorithm
  algorithm_factory
)
