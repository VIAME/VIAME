/*ckwg +29
 * Copyright 2015-2018 by Kitware, Inc.
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

/**
 * \file
 * \brief Implementation of enum to/from string conversions
 */


#include <arrows/ceres/types.h>
#include <arrows/ceres/lens_distortion.h>
#include <ceres/loss_function.h>



using namespace kwiver::vital;


#define CERES_ENUM_HELPERS(NS, ceres_type)                              \
namespace kwiver {                                                      \
namespace vital {                                                       \
                                                                        \
template<>                                                              \
config_block_value_t                                                    \
config_block_set_value_cast(NS::ceres_type const& value)                \
{                                                                       \
  return NS::ceres_type##ToString(value);                               \
}                                                                       \
                                                                        \
template<>                                                              \
NS::ceres_type                                                          \
config_block_get_value_cast(config_block_value_t const& value)          \
{                                                                       \
  NS::ceres_type cet;                                                   \
  if(!NS::StringTo##ceres_type(value, &cet))                            \
  {                                                                     \
    VITAL_THROW( bad_config_block_cast,value);                          \
  }                                                                     \
  return cet;                                                           \
}                                                                       \
                                                                        \
}                                                                       \
                                                                        \
namespace arrows {                                                      \
namespace ceres {                                                       \
                                                                        \
template<>                                                              \
std::string                                                             \
ceres_options< NS::ceres_type >()                                       \
{                                                                       \
  typedef NS::ceres_type T;                                             \
  std::string options_str = "\nMust be one of the following options:";  \
  std::string opt;                                                      \
  for (unsigned i=0; i<20; ++i)                                         \
  {                                                                     \
    opt = NS::ceres_type##ToString(static_cast<T>(i));                  \
    if (opt == "UNKNOWN")                                               \
    {                                                                   \
      break;                                                            \
    }                                                                   \
    options_str += "\n  - " + opt;                                      \
  }                                                                     \
  return options_str;                                                   \
}                                                                       \
                                                                        \
}                                                                       \
}                                                                       \
}

CERES_ENUM_HELPERS(::ceres, LinearSolverType)
CERES_ENUM_HELPERS(::ceres, PreconditionerType)
CERES_ENUM_HELPERS(::ceres, TrustRegionStrategyType)
CERES_ENUM_HELPERS(::ceres, DoglegType)

CERES_ENUM_HELPERS(kwiver::arrows::ceres, LossFunctionType)
CERES_ENUM_HELPERS(kwiver::arrows::ceres, LensDistortionType)
CERES_ENUM_HELPERS(kwiver::arrows::ceres, CameraIntrinsicShareType)

#undef CERES_ENUM_HELPERS


namespace kwiver {
namespace arrows {
namespace ceres {

#define CASESTR(x) case x: return #x
#define STRENUM(x) if (value == #x) { *type = x; return true;}

/// Convert a string to uppercase
static void UpperCase(std::string* input)
{
  std::transform(input->begin(), input->end(), input->begin(), ::toupper);
}


/// Provide a string representation for a LossFunctionType value
const char*
LossFunctionTypeToString(LossFunctionType type)
{
  switch (type)
  {
    CASESTR(TRIVIAL_LOSS);
    CASESTR(HUBER_LOSS);
    CASESTR(SOFT_L_ONE_LOSS);
    CASESTR(CAUCHY_LOSS);
    CASESTR(ARCTAN_LOSS);
    CASESTR(TUKEY_LOSS);
    default:
      return "UNKNOWN";
  }
}


/// Parse a LossFunctionType value from a string or return false
bool
StringToLossFunctionType(std::string value, LossFunctionType* type)
{
  UpperCase(&value);
  STRENUM(TRIVIAL_LOSS);
  STRENUM(HUBER_LOSS);
  STRENUM(SOFT_L_ONE_LOSS);
  STRENUM(CAUCHY_LOSS);
  STRENUM(ARCTAN_LOSS);
  STRENUM(TUKEY_LOSS);
  return false;
}


/// Provide a string representation for a LensDisortionType value
const char*
LensDistortionTypeToString(LensDistortionType type)
{
  switch (type)
  {
    CASESTR(NO_DISTORTION);
    CASESTR(POLYNOMIAL_RADIAL_DISTORTION);
    CASESTR(POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION);
    CASESTR(RATIONAL_RADIAL_TANGENTIAL_DISTORTION);
    default:
      return "UNKNOWN";
  }
}


/// Parse a LensDistortionType value from a string or return false
bool
StringToLensDistortionType(std::string value, LensDistortionType* type)
{
  UpperCase(&value);
  STRENUM(NO_DISTORTION);
  STRENUM(POLYNOMIAL_RADIAL_DISTORTION);
  STRENUM(POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION);
  STRENUM(RATIONAL_RADIAL_TANGENTIAL_DISTORTION);
  return false;
}


/// Provide a string representation for a CameraIntrinsicShareType value
KWIVER_ALGO_CERES_EXPORT const char*
CameraIntrinsicShareTypeToString(CameraIntrinsicShareType type)
{
  switch (type)
  {
    CASESTR(AUTO_SHARE_INTRINSICS);
    CASESTR(FORCE_COMMON_INTRINSICS);
    CASESTR(FORCE_UNIQUE_INTRINSICS);
    default:
      return "UNKNOWN";
  }
}


/// Parse a CameraIntrinsicShareType value from a string or return false
KWIVER_ALGO_CERES_EXPORT bool
StringToCameraIntrinsicShareType(std::string value, CameraIntrinsicShareType* type)
{
  UpperCase(&value);
  STRENUM(AUTO_SHARE_INTRINSICS);
  STRENUM(FORCE_COMMON_INTRINSICS);
  STRENUM(FORCE_UNIQUE_INTRINSICS);
  return false;
}


#undef CASESTR
#undef STRENUM


/// Construct a LossFunction object from the specified enum type
::ceres::LossFunction*
LossFunctionFactory(LossFunctionType type, double s)
{
  switch(type)
  {
    case TRIVIAL_LOSS:
      return NULL;
    case HUBER_LOSS:
      return new ::ceres::HuberLoss(s);
    case SOFT_L_ONE_LOSS:
      return new ::ceres::SoftLOneLoss(s);
    case CAUCHY_LOSS:
      return new ::ceres::CauchyLoss(s);
    case ARCTAN_LOSS:
      return new ::ceres::ArctanLoss(s);
    case TUKEY_LOSS:
      return new ::ceres::TukeyLoss(s);
    default:
      return NULL;
  }
}

/// Return the number of distortion parameters required for each type
unsigned int
num_distortion_params(LensDistortionType type)
{
  switch(type)
  {
  case POLYNOMIAL_RADIAL_DISTORTION:
    return distortion_poly_radial::num_coeffs;
  case POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION:
    return distortion_poly_radial_tangential::num_coeffs;
  case RATIONAL_RADIAL_TANGENTIAL_DISTORTION:
    return distortion_ratpoly_radial_tangential::num_coeffs;
  default:
    return 0;
  }
}

} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver
