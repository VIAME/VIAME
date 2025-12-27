/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_ITK_REGISTER_OPTICAL_AND_THERMAL_H
#define VIAME_ITK_REGISTER_OPTICAL_AND_THERMAL_H

#include <plugins/itk/viame_itk_export.h>

#include "itkImage.h"
#include "itkAffineTransform.h"
#include "itkCompositeTransform.h"

namespace viame
{

namespace itk
{

constexpr unsigned int Dimension = 2;

using TransformFloatType = double;

using OpticalPixelType = unsigned short;
using ThermalPixelType = unsigned short;
using WarpedOpticalPixelType = unsigned char;
using WarpedThermalPixelType = unsigned short;

using OpticalImageType = ::itk::Image< OpticalPixelType, Dimension >;
using ThermalImageType = ::itk::Image< ThermalPixelType, Dimension >;
using WarpedOpticalImageType = ::itk::Image< WarpedOpticalPixelType, Dimension >;
using WarpedThermalImageType = ::itk::Image< WarpedThermalPixelType, Dimension >;

using BaseTransformType = ::itk::Transform< TransformFloatType, Dimension, Dimension >;
using AffineTransformType = ::itk::AffineTransform< TransformFloatType, Dimension >;
using NetTransformType = ::itk::CompositeTransform< TransformFloatType, Dimension >;

VIAME_ITK_EXPORT bool PerformRegistration(
  const OpticalImageType& inputOpticalImage,
  const ThermalImageType& inputThermalImage,
  NetTransformType::Pointer& outputTransformation,
  const double opticalImageShrinkFactor = 10.0,
  const double thermalImageShrinkFactor = 1.0,
  const unsigned numberOfIterations = 100,
  const double maximumPhysicalStepSize = 2.0,
  const double pointSetSigma = 3.0 );

VIAME_ITK_EXPORT bool WarpThermalToOpticalImage(
  const OpticalImageType& inputOpticalImage,
  const ThermalImageType& inputThermalImage,
  const NetTransformType& inputTransformation,
  WarpedThermalImageType::Pointer& outputWarpedImage );

VIAME_ITK_EXPORT bool WarpThermalToOpticalImage(
  const ThermalImageType& inputThermalImage,
  const NetTransformType& inputTransformation,
  const ThermalImageType::SizeType& outputSize,
  WarpedThermalImageType::Pointer& outputWarpedImage );

VIAME_ITK_EXPORT bool WarpOpticalToThermalImage(
  const OpticalImageType& inputOpticalImage,
  const ThermalImageType& inputThermalImage,
  const NetTransformType& inputTransformation,
  WarpedOpticalImageType::Pointer& outputWarpedImage );

} // end namespace itk

} // end namespace viame

#endif // VIAME_ITK_REGISTER_OPTICAL_AND_THERMAL_H
