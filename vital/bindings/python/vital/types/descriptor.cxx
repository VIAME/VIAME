/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <pybind11/stl.h>

#include <vital/types/descriptor.h>

namespace py = pybind11;

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
