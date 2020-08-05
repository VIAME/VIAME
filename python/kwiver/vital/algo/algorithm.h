#ifndef KWIVER_VITAL_PYTHON_ALGORITHM_H_
#define KWIVER_VITAL_PYTHON_ALGORITHM_H_

#include <pybind11/pybind11.h>
namespace py = pybind11;

void algorithm(py::module &m);

template<class implementation, class trampoline>
void register_algorithm(py::module &m,
                        const std::string implementation_name)
{
  std::stringstream impl_name;
  impl_name << "_algorithm<" << implementation_name << ">";

  py::class_< kwiver::vital::algorithm_def<implementation>,
              std::shared_ptr<kwiver::vital::algorithm_def<implementation>>,
              kwiver::vital::algorithm,
              trampoline >(m, impl_name.str().c_str())
    .def(py::init())
    .def_static("create", &kwiver::vital::algorithm_def<implementation>::create)
    .def_static("registered_names",
                &kwiver::vital::algorithm_def<implementation>::registered_names)
    .def_static("get_nested_algo_configuration",
                &kwiver::vital::algorithm_def<implementation>::get_nested_algo_configuration)
    .def_static("set_nested_algo_configuration",
                &kwiver::vital::algorithm_def<implementation>::set_nested_algo_configuration)
    .def_static("check_nested_algo_configuration",
                &kwiver::vital::algorithm_def<implementation>::check_nested_algo_configuration);
}
#endif
