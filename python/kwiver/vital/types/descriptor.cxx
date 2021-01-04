// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/stl.h>

#include <vital/types/descriptor.h>

namespace py = pybind11;
namespace kwiver {
namespace vital  {
namespace python {
py::object
new_descriptor(size_t len, char ctype)
{
  py::object retVal;
  if(ctype == 'd')
  {
    auto obj = std::shared_ptr<kwiver::vital::descriptor_dynamic<double>>(new kwiver::vital::descriptor_dynamic<double>(len));
    retVal = py::cast<std::shared_ptr<kwiver::vital::descriptor_dynamic<double>>>(obj);
  }
  else if(ctype == 'f')
  {
    auto obj = std::shared_ptr<kwiver::vital::descriptor_dynamic<float>>(new kwiver::vital::descriptor_dynamic<float>(len));
    retVal = py::cast<std::shared_ptr<kwiver::vital::descriptor_dynamic<float>>>(obj);
  }
  return retVal;
}

double
sum_descriptors(std::shared_ptr<kwiver::vital::descriptor> &desc)
{
  std::vector<double> doubles = desc->as_double();
  double sum = 0;
  for(auto val : doubles)
  {
    sum += val;
  }
  return sum;
}

template<class T>
void
set_slice(std::shared_ptr<kwiver::vital::descriptor_dynamic<T>> self, py::slice slice, py::object val_obj)
{
  size_t start, stop, step, slicelength;
  slice.compute(self->size(), &start, &stop, &step, &slicelength);
  T* data = self->raw_data();
  try
  {
    T val = val_obj.cast<T>();

    for (size_t idx = start; idx < stop; idx+=step)
    {
      data[idx] = val;
    }
  }
  catch(...)
  {
    std::vector<T> val = val_obj.cast<std::vector<T>>();

    for (size_t idx = start; idx < stop; idx+=step)
    {
      data[idx] = val[idx]; // if there's an out of bounds here, python will throw an exception
    }
  }
}

template<class T>
void
set_index(std::shared_ptr<kwiver::vital::descriptor_dynamic<T>> self, size_t idx, T val)
{
  T* data = self->raw_data();
  data[idx] = val;
}

template<class T>
std::vector<T>
get_slice(std::shared_ptr<kwiver::vital::descriptor_dynamic<T>> self, py::slice slice)
{
  std::vector<T> ret_vec;
  size_t start, stop, step, slicelength;
  slice.compute(self->size(), &start, &stop, &step, &slicelength);
  T* data = self->raw_data();

  for (size_t idx = start; idx < stop; idx+=step)
  {
    ret_vec.push_back(data[idx]);
  }
  return ret_vec;
}

template<class T>
T
get_index(std::shared_ptr<kwiver::vital::descriptor_dynamic<T>> self, size_t idx)
{
  T* data = self->raw_data();
  return data[idx];
}
}
}
}

using namespace kwiver::vital::python;
PYBIND11_MODULE(descriptor, m)
{
  // we have to use a separate function to initialize Descriptors, because it can return one of two separate types (DescriptorD or DescriptorF)
  m.def("new_descriptor", &new_descriptor,
    py::arg("size")=0, py::arg("ctype")='d');

  // everything we can fit in the parent class goes there
  py::class_<kwiver::vital::descriptor, std::shared_ptr<kwiver::vital::descriptor>>(m, "Descriptor")
  .def("sum", &sum_descriptors)
  .def("todoublearray", &kwiver::vital::descriptor::as_double)
  .def("tobytearray", &kwiver::vital::descriptor::as_bytes)
  .def("__eq__", &kwiver::vital::descriptor::operator==)
  .def("__ne__", &kwiver::vital::descriptor::operator!=)
  .def_property_readonly("size", &kwiver::vital::descriptor::size)
  .def_property_readonly("nbytes", &kwiver::vital::descriptor::num_bytes)
  ;

  // Because slices need to use the raw_data function, we can't use kwiver::vital::descriptor
  py::class_<kwiver::vital::descriptor_dynamic<double>, kwiver::vital::descriptor, std::shared_ptr<kwiver::vital::descriptor_dynamic<double>>>(m, "DescriptorD")
  .def("__setitem__", &set_slice<double>,
    py::arg("slice"), py::arg("value"))
  .def("__getitem__", &get_slice<double>,
    py::arg("slice"))
  .def("__setitem__", &set_index<double>,
    py::arg("index"), py::arg("value"))
  .def("__getitem__", &get_index<double>,
    py::arg("index"));
  py::class_<kwiver::vital::descriptor_dynamic<float>, kwiver::vital::descriptor, std::shared_ptr<kwiver::vital::descriptor_dynamic<float>>>(m, "DescriptorF")
  .def("__setitem__", &set_slice<float>,
    py::arg("slice"), py::arg("value"))
  .def("__getitem__", &get_slice<float>,
    py::arg("slice"))
  .def("__setitem__", &set_index<float>,
    py::arg("index"), py::arg("value"))
  .def("__getitem__", &get_index<float>,
    py::arg("index"));
}
