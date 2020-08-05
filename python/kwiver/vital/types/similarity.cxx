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

#include <vital/types/similarity.h>

#include "rotation_class.cxx"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

typedef kwiver::vital::similarity_<float> similarity_f;
typedef kwiver::vital::similarity_<double> similarity_d;

class PySimilarityBase
{

  public:
    virtual ~PySimilarityBase() = default;

    virtual py::object get_scale() {return py::none(); };
    virtual PyRotation get_rotation() {return PyRotation(); };
    virtual py::object get_translation() {return py::none(); };

    virtual py::object get_similarity() {return py::none(); };

    virtual py::object as_matrix() {return py::none(); };

    virtual std::shared_ptr<PySimilarityBase> compose(std::shared_ptr<PySimilarityBase>)
      {return std::shared_ptr<PySimilarityBase>(new PySimilarityBase());};

    virtual std::shared_ptr<PySimilarityBase> inverse()
      {return std::shared_ptr<PySimilarityBase>();};

    virtual py::object transform_vector(py::object) { return py::none(); };

    virtual bool operator==(std::shared_ptr<PySimilarityBase>) { return false; };
    bool operator!=(std::shared_ptr<PySimilarityBase> other) { return !(*this==other); };
};

class PySimilarityD
: public PySimilarityBase
{
  similarity_d similarity;

  public:

    PySimilarityD(double scale,
                  kwiver::vital::rotation_<double> rotation,
                  Eigen::Matrix<double,3,1> trans)
                  : similarity(scale,rotation,trans) {};
    PySimilarityD(Eigen::Matrix<double,4,4> mat)
                  : similarity(mat) {};
    PySimilarityD(similarity_d similarity)
                  : similarity(similarity) {};

    py::object get_scale() {return py::cast<double>(similarity.scale()); };
    PyRotation get_rotation()
                           {
                             PyRotation rot('d');
                             rot.setRotD(similarity.rotation());
                             return rot;
                           };
    py::object get_translation() {return py::cast<Eigen::Matrix<double,3,1>>(similarity.translation()); };

    py::object as_matrix() {return py::cast<Eigen::Matrix<double,4,4>>(similarity.matrix()); };

    std::shared_ptr<PySimilarityBase> inverse()
      {
        return std::shared_ptr<PySimilarityBase>(new PySimilarityD(this->similarity.inverse()));
      }

    std::shared_ptr<PySimilarityBase> compose(std::shared_ptr<PySimilarityBase> other)
      {
        auto other_sim = similarity_d(other->as_matrix().cast<Eigen::Matrix<double,4,4>>());
        auto sim = new PySimilarityD(this->similarity * other_sim);
        return std::shared_ptr<PySimilarityBase>(sim);
      };

    py::object transform_vector(py::object);

    bool operator==(std::shared_ptr<PySimilarityBase> other) { return similarity.matrix() == other->as_matrix().cast<Eigen::Matrix<double,4,4>>();};
};

class PySimilarityF
: public PySimilarityBase
{
  similarity_f similarity;

  public:

    PySimilarityF(float scale,
                  kwiver::vital::rotation_<float> rotation,
                  Eigen::Matrix<float,3,1> trans)
                  : similarity(scale,rotation,trans) {};
    PySimilarityF(Eigen::Matrix<float,4,4> mat)
                  : similarity(mat) {};
    PySimilarityF(similarity_f similarity)
                  : similarity(similarity) {};


    py::object get_scale() {return py::cast<float>(similarity.scale()); };
    PyRotation get_rotation()
                           {
                             PyRotation rot('f');
                             rot.setRotF(similarity.rotation());
                             return rot;
                           };
    py::object get_translation() {return py::cast<Eigen::Matrix<float,3,1>>(similarity.translation()); };

    py::object as_matrix() {return py::cast<Eigen::Matrix<float,4,4>>(similarity.matrix()); };

    std::shared_ptr<PySimilarityBase> inverse()
      {
        return std::shared_ptr<PySimilarityBase>(new PySimilarityF(this->similarity.inverse()));
      }

    std::shared_ptr<PySimilarityBase> compose(std::shared_ptr<PySimilarityBase> other)
      {
        auto other_sim = similarity_f(other->as_matrix().cast<Eigen::Matrix<float,4,4>>());
        auto sim = new PySimilarityF(this->similarity * other_sim);
        return std::shared_ptr<PySimilarityBase>(sim);
      };

    py::object transform_vector(py::object);

    bool operator==(std::shared_ptr<PySimilarityBase> other) { return similarity.matrix() == other->as_matrix().cast<Eigen::Matrix<float,4,4>>();};
};

py::object
PySimilarityD::
transform_vector(py::object other)
{
  Eigen::Matrix<double,3,1> mat = other.cast<Eigen::Matrix<double,3,1>>();
  mat = similarity*mat;
  return py::cast<Eigen::Matrix<double,3,1>>(mat);
}

py::object
PySimilarityF::
transform_vector(py::object other)
{
  Eigen::Matrix<float,3,1> mat = other.cast<Eigen::Matrix<float,3,1>>();
  mat = similarity*mat;
  return py::cast<Eigen::Matrix<float,3,1>>(mat);
}

std::shared_ptr<PySimilarityBase>
new_similarity(py::object scale_obj,
               std::shared_ptr<PyRotation> rot,
               py::object trans_obj,
               char type)
{
  std::shared_ptr<PySimilarityBase> retVal;
  if(type == 'd')
  {
    double scale = scale_obj.cast<double>();
    auto rotation_mat = rot->matrix().cast<Eigen::Matrix<double,3,3>>();
    kwiver::vital::rotation_<double> rotation(rotation_mat);
    auto trans = trans_obj.cast<Eigen::Matrix<double,3,1>>();
    retVal = std::shared_ptr<PySimilarityBase>(new PySimilarityD(scale, rotation, trans));
  }
  else if(type == 'f')
  {
    float scale = scale_obj.cast<float>();
    auto rotation_mat = rot->matrix().cast<Eigen::Matrix<float,3,3>>();
    kwiver::vital::rotation_<float> rotation(rotation_mat);
    auto trans = trans_obj.cast<Eigen::Matrix<float,3,1>>();
    retVal = std::shared_ptr<PySimilarityBase>(new PySimilarityF(scale, rotation, trans));
  }
  return retVal;
}

std::shared_ptr<PySimilarityBase>
new_similarity_from_matrix(py::object mat_obj)
{
  std::shared_ptr<PySimilarityBase> retVal;
  try
  {
    auto mat = mat_obj.cast<Eigen::Matrix<double,4,4>>();
    retVal = std::shared_ptr<PySimilarityBase>(new PySimilarityD(mat));
  } catch(...)
  {
    auto mat = mat_obj.cast<Eigen::Matrix<float,4,4>>();
    retVal = std::shared_ptr<PySimilarityBase>(new PySimilarityF(mat));
  }
  return retVal;
}


PYBIND11_MODULE(similarity, m)
{
  py::class_<PySimilarityBase, std::shared_ptr<PySimilarityBase>>(m, "Similarity")
  .def(py::init(&new_similarity),
    py::arg("scale")=1., py::arg("rotation")=PyRotation(), py::arg("translation")=Eigen::Matrix<double,3,1>::Zero(), py::arg("ctype")='d')
  .def_static("from_matrix", &new_similarity_from_matrix,
    py::arg("matrix"))
  .def("as_matrix", &PySimilarityBase::as_matrix)
  .def("compose", &PySimilarityBase::compose,
    py::arg("other"))
  .def("inverse", &PySimilarityBase::inverse)
  .def("transform_vector", &PySimilarityBase::transform_vector)
  .def("__eq__", &PySimilarityBase::operator==,
    py::arg("other"))
  .def("__ne__", &PySimilarityBase::operator!=,
    py::arg("other"))
  .def("__mul__", &PySimilarityBase::compose,
    py::arg("other"))
  .def("__mul__", &PySimilarityBase::transform_vector,
    py::arg("other"))
  .def_property_readonly("scale", &PySimilarityBase::get_scale)
  .def_property_readonly("rotation", &PySimilarityBase::get_rotation)
  .def_property_readonly("translation", &PySimilarityBase::get_translation)
  ;
}
