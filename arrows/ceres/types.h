// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
