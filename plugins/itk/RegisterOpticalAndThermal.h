/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

using AffineTransformType = ::itk::AffineTransform< TransformFloatType, Dimension >;
using NetTransformType = ::itk::CompositeTransform< TransformFloatType, Dimension >;

VIAME_ITK_EXPORT bool PerformRegistration(
  const OpticalImageType& inputOpticalImage,
  const ThermalImageType& inputThermalImage,
  NetTransformType::Pointer& outputTransformation,
  const double opticalImageShrinkFactor = 10.0,
  const double thermalImageShrinkFactor = 1.0,
  const unsigned iterationCount = 100,
  const double maximumPhysicalStepSize = 2.0,
  const double pointSetSigma = 3.0 );

VIAME_ITK_EXPORT bool WarpThermalToOpticalImage(
  const OpticalImageType& inputOpticalImage,
  const ThermalImageType& inputThermalImage,
  const NetTransformType& inputTransformation,
  WarpedThermalImageType::Pointer& outputWarpedImage );

VIAME_ITK_EXPORT bool WarpOpticalToThermalImage(
  const OpticalImageType& inputOpticalImage,
  const ThermalImageType& inputThermalImage,
  const NetTransformType& inputTransformation,
  WarpedOpticalImageType::Pointer& outputWarpedImage );

} // end namespace itk

} // end namespace viame

#endif // VIAME_ITK_REGISTER_OPTICAL_AND_THERMAL_H
