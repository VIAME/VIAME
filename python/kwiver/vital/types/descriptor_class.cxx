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

#include <vital/types/descriptor.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyDescriptorBase
{

  public:

    virtual ~PyDescriptorBase() = default;

    virtual size_t get_size() {return 0;};
    virtual size_t get_num_bytes() {return 0;};
    virtual size_t sum() {return -1;};

    virtual void set_slice(py::slice slice, py::object val_obj) {};
    virtual py::object get_slice(py::slice slice) {return py::none();};

    virtual std::vector<double> as_double() { return std::vector<double>();};
    virtual kwiver::vital::byte const* as_bytes() { return nullptr;};

};

class PyDescriptorD
: public PyDescriptorBase
{

  kwiver::vital::descriptor_dynamic<double> desc;

  public:

    PyDescriptorD(size_t len)
      : desc (kwiver::vital::descriptor_dynamic<double>(len))
      {};

    size_t get_size() { return desc.size(); };
    size_t get_num_bytes() { return desc.num_bytes(); };

    size_t sum()
    {
      double* data = desc.raw_data();
      size_t sum = 0;
      for (size_t idx = 0; idx < desc.size(); idx++)
      {
        sum += data[idx];
      }
      return sum;
    }

    std::vector<double> as_double() { return desc.as_double();};
    kwiver::vital::byte const* as_bytes() { return desc.as_bytes();};

    void set_slice(py::slice slice, py::object val_obj)
    {
      size_t start, stop, step, slicelength;
      slice.compute(desc.size(), &start, &stop, &step, &slicelength);
      double* data = desc.raw_data();

      try
      {
        double val = val_obj.cast<double>();

        for (size_t idx = start; idx < stop; idx+=step)
        {
          data[idx] = val;
        }
      }
      catch(...)
      {
        std::vector<double> val = val_obj.cast<std::vector<double>>();

        for (size_t idx = start; idx < stop; idx+=step)
        {
          data[idx] = val[idx];
        }
      }
    };

    py::object get_slice(py::slice slice)
    {
      std::vector<double> ret_vec;
      size_t start, stop, step, slicelength;
      slice.compute(desc.size(), &start, &stop, &step, &slicelength);
      double* data = desc.raw_data();

      for (size_t idx = start; idx < stop; idx+=step)
      {
        ret_vec.push_back(data[idx]);
      }
      return py::cast<std::vector<double>> (ret_vec);
    }

};

class PyDescriptorF
: public PyDescriptorBase
{

  kwiver::vital::descriptor_dynamic<float> desc;

  public:

    PyDescriptorF(size_t len)
      : desc(kwiver::vital::descriptor_dynamic<float>(len))
      {};

    size_t get_size() { return desc.size(); };
    size_t get_num_bytes() { return desc.num_bytes(); };

    size_t sum()
    {
      float* data = desc.raw_data();
      size_t sum = 0;
      for (size_t idx = 0; idx < desc.size(); idx++)
      {
        sum += data[idx];
      }
      return sum;
    }

    std::vector<double> as_double() { return desc.as_double();};
    kwiver::vital::byte const* as_bytes() { return desc.as_bytes();};

    void set_slice(py::slice slice, py::object val_obj)
    {
      size_t start, stop, step, slicelength;
      slice.compute(desc.size(), &start, &stop, &step, &slicelength);
      float* data = desc.raw_data();

      try
      {
        float val = val_obj.cast<float>();

        for (size_t idx = start; idx < stop; idx+=step)
        {
          data[idx] = val;
        }
      }
      catch(...)
      {
        std::vector<float> val = val_obj.cast<std::vector<float>>();

        for (size_t idx = start; idx < stop; idx+=step)
        {
          data[idx] = val[idx];
        }
      }
    };

    py::object get_slice(py::slice slice)
    {
      std::vector<float> ret_vec;
      size_t start, stop, step, slicelength;
      slice.compute(desc.size(), &start, &stop, &step, &slicelength);
      float* data = desc.raw_data();

      for (size_t idx = start; idx < stop; idx+=step)
      {
        ret_vec.push_back(data[idx]);
      }
      return py::cast<std::vector<float>> (ret_vec);
    }
};

std::shared_ptr<PyDescriptorBase>
new_descriptor(size_t len, char ctype)
{
  std::shared_ptr<PyDescriptorBase> retVal;
  if(ctype == 'd')
  {
    retVal = std::shared_ptr<PyDescriptorBase>(new PyDescriptorD(len));
  }
  else if(ctype == 'f')
  {
    retVal = std::shared_ptr<PyDescriptorBase>(new PyDescriptorF(len));
  }
  return retVal;
}

class PyDescriptorSet
{
  std::vector<std::shared_ptr<PyDescriptorBase>> descriptors;

  public:

    PyDescriptorSet() {};
    PyDescriptorSet(py::list desc_arg)
     {
       for(auto py_desc: desc_arg)
       {
         std::shared_ptr<PyDescriptorBase> desc = py_desc.cast<std::shared_ptr<PyDescriptorBase>>();
         descriptors.push_back(desc);
       }
     };

    size_t size() { return descriptors.size(); };

    std::vector<std::shared_ptr<PyDescriptorBase>> get_descriptors() { return descriptors; };
};
