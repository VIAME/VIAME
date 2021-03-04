/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
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

#include <vital/types/transform_2d.h>

#include <python/kwiver/vital/util/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py=pybind11;
namespace kv=kwiver::vital;

class transform_2d_trampoline
  : public kv::transform_2d
{
public:
  using kv::transform_2d::transform_2d;
  kv::transform_2d_sptr clone() const override;
  kv::vector_2d map( kv::vector_2d const& p ) const override;
  kv::transform_2d_sptr inverse_() const override;
};

// Lets python classes override this virtual function
class transform_2d_publicist
  : public kv::transform_2d
{
public:
  using kv::transform_2d::inverse_;
};
PYBIND11_MODULE(transform_2d, m)
{
  py::class_<kv::transform_2d, transform_2d_trampoline,
    std::shared_ptr<kv::transform_2d>>(m, "Transform2D")
  .def(py::init<>())
  .def("map", &kv::transform_2d::map)
  .def("inverse", &kv::transform_2d::inverse)
  ;
}
// We are excluding clone and inverse_ in the base's binding code to follow the pattern
// described in this pybind issue:
// https://github.com/pybind/pybind11/issues/1049#issuecomment-326688270
// Subclasses will still be able to override it, however.
// Pybind automatically downcasts pointers returned by clone and inverse_ to the lowest possible
// subtype, but under certain circumstances, the returned pointer can get sliced.
// The above link has a solution to this issue. The trampoline's implementation of
// clone and inverse_ also was modified to follow this pattern

// NOTE: Since inverse was public, it will need to be defined in any subclass' binding code,
// assuming that the subclass implements inverse_. Homography implements its own inverse() that
// has a different return type, which is adequate.

kv::transform_2d_sptr
transform_2d_trampoline
::clone() const
{
  auto self = py::cast(this);
  auto cloned = self.attr("clone")();

  auto keep_python_state_alive = std::make_shared<py::object>(cloned);
  auto ptr = cloned.cast<transform_2d_trampoline*>();

  // aliasing shared_ptr: points to `transform_2d_trampoline* ptr`
  // but refcounts the Python object
  return std::shared_ptr<kv::transform_2d>(keep_python_state_alive, ptr);
}

kv::vector_2d
transform_2d_trampoline
::map( kv::vector_2d const& p ) const
{
  VITAL_PYBIND11_OVERLOAD_PURE(
    kv::vector_2d,
    kv::transform_2d,
    map,
    p
  );
}

kv::transform_2d_sptr
transform_2d_trampoline
::inverse_() const
{
  auto self = py::cast(this);
  auto inverse_inst = self.attr("inverse_")();

  auto keep_python_state_alive = std::make_shared<py::object>(inverse_inst);
  auto ptr = inverse_inst.cast<transform_2d_trampoline*>();

  // aliasing shared_ptr: points to `transform_2d_trampoline* ptr`
  // but refcounts the Python object
  return std::shared_ptr<kv::transform_2d>(keep_python_state_alive, ptr);
}
