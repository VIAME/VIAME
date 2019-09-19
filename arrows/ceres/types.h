/*ckwg +29
 * Copyright 2015-2016, 2019 by Kitware, Inc.
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
 * \brief Define additional enum types in a similar style as Ceres
 */

#ifndef KWIVER_ARROWS_CERES_TYPES_H_
#define KWIVER_ARROWS_CERES_TYPES_H_


#include <vital/vital_config.h>
#include <vital/config/config_block.h>
#include <arrows/ceres/kwiver_algo_ceres_export.h>

#include <string>
#include <ceres/ceres.h>

namespace kwiver {
namespace arrows {
namespace ceres {

/// The various robust loss function supported in the config
enum LossFunctionType
{
  TRIVIAL_LOSS,
  HUBER_LOSS,
  SOFT_L_ONE_LOSS,
  CAUCHY_LOSS,
  ARCTAN_LOSS,
  TUKEY_LOSS
};

/// The various models for lens distortion supported in the config
enum LensDistortionType
{
  NO_DISTORTION,
  POLYNOMIAL_RADIAL_DISTORTION,
  POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION,
  RATIONAL_RADIAL_TANGENTIAL_DISTORTION
};

/// The options for camera intrinsic sharing supported in the config
enum CameraIntrinsicShareType
{
  AUTO_SHARE_INTRINSICS,
  FORCE_COMMON_INTRINSICS,
  FORCE_UNIQUE_INTRINSICS
};


/// Provide a string representation for a LossFunctionType value
KWIVER_ALGO_CERES_EXPORT const char*
LossFunctionTypeToString(LossFunctionType type);

/// Parse a LossFunctionType value from a string or return false
KWIVER_ALGO_CERES_EXPORT bool
StringToLossFunctionType(std::string value, LossFunctionType* type);

/// Construct a LossFunction object from the specified enum type
KWIVER_ALGO_CERES_EXPORT ::ceres::LossFunction*
LossFunctionFactory(LossFunctionType type, double scale=1.0);


/// Provide a string representation for a LensDisortionType value
KWIVER_ALGO_CERES_EXPORT const char*
LensDistortionTypeToString(LensDistortionType type);

/// Parse a LensDistortionType value from a string or return false
KWIVER_ALGO_CERES_EXPORT bool
StringToLensDistortionType(std::string value, LensDistortionType* type);

/// Return the number of distortion parameters required for each type
KWIVER_ALGO_CERES_EXPORT unsigned int
num_distortion_params(LensDistortionType type);


/// Provide a string representation for a CameraIntrinsicShareType value
KWIVER_ALGO_CERES_EXPORT const char*
CameraIntrinsicShareTypeToString(CameraIntrinsicShareType type);

/// Parse a CameraIntrinsicShareType value from a string or return false
KWIVER_ALGO_CERES_EXPORT bool
StringToCameraIntrinsicShareType(std::string value, CameraIntrinsicShareType* type);


/// Defult implementation of string options for Ceres enums
template <typename T>
std::string
ceres_options()
{
  return std::string();
}

} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver


#define CERES_ENUM_HELPERS(NS, ceres_type)                              \
namespace kwiver {                                                      \
namespace vital {                                                       \
                                                                        \
template<>                                                              \
config_block_value_t                                                    \
config_block_set_value_cast(NS::ceres_type const& value);               \
                                                                        \
template<>                                                              \
NS::ceres_type                                                          \
config_block_get_value_cast(config_block_value_t const& value);         \
                                                                        \
}                                                                       \
                                                                        \
namespace arrows {                                                      \
namespace ceres {                                                       \
                                                                        \
template<>                                                              \
std::string                                                             \
ceres_options< NS::ceres_type >();                                      \
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


#endif
