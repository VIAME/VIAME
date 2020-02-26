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

#include <vital/types/covariance.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <python/kwiver/vital/types/covariance_class.h>

// Unfortunately, because pybind11 can't deal with templates, we have to enumerate
// all the possibilities. But by handling it at construction time, we don't need
// to worry about this later.
std::shared_ptr<PyCovarianceBase>
PyCovarianceBase::
covar_from_scalar(int N, char c_type, py::object init)
{
  std::shared_ptr<PyCovarianceBase> retVal;
  if(N == 2 && c_type == 'd')
  {
    if(!init.is(py::none()))
    {
      double mat = init.cast< double >();
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2d(mat));
    }
    else
    {
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2d());
    }
  }
  else if(N == 3 && c_type == 'd')
  {
    if(!init.is(py::none()))
    {
      double mat = init.cast< double >();
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3d(mat));
    }
    else
    {
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3d());
    }
  }
  else if(N == 2 && c_type == 'f')
  {
    if(!init.is(py::none()))
    {
      float mat = init.cast< float >();
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2f(mat));
    }
    else
    {
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2f());
    }
  }
  else if(N == 3 && c_type == 'f')
  {
    if(!init.is(py::none()))
    {
      float mat = init.cast< float >();
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3f(mat));
    }
    else
    {
      retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3f());
    }
  }
  return retVal;
}

std::shared_ptr<PyCovarianceBase>
PyCovarianceBase::
covar_from_matrix(int N, char c_type, py::object init)
{
  std::shared_ptr<PyCovarianceBase> retVal;
  if(N == 2 && c_type == 'd')
  {
    Eigen::Matrix< double, 2, 2 > mat = init.cast< Eigen::Matrix< double, 2, 2 > >();
    retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2d(mat));
  }
  else if(N == 3 && c_type == 'd')
  {
    Eigen::Matrix< double, 3, 3 > mat = init.cast< Eigen::Matrix< double, 3, 3 > >();
    retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3d(mat));
  }
  else if(N == 2 && c_type == 'f')
  {
    Eigen::Matrix< float, 2, 2 > mat = init.cast< Eigen::Matrix< float, 2, 2 > >();
    retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance2f(mat));
  }
  else if(N == 3 && c_type == 'f')
  {
    Eigen::Matrix< float, 3, 3 > mat = init.cast< Eigen::Matrix< float, 3, 3 > >();
    retVal = std::shared_ptr<PyCovarianceBase>(new PyCovariance3f(mat));
  }
  return retVal;
}
