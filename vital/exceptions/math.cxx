// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation for math exceptions.
 */

#include "math.h"

namespace kwiver {
namespace vital {

math_exception
::math_exception() noexcept
{
  m_what = "A math exception occurred.";
}

math_exception
::~math_exception() noexcept
{
}

non_invertible
::non_invertible() noexcept
{
  m_what = "A transformation was found to be non-invertible";
}

non_invertible
::~non_invertible() noexcept
{
}

point_maps_to_infinity
::point_maps_to_infinity() noexcept
{
  m_what = "A point mapped to infinity";
}

point_maps_to_infinity
::~point_maps_to_infinity() noexcept
{
}

invalid_matrix_operation
::invalid_matrix_operation(std::string reason) noexcept
{
  m_what = "Invalid operation: " + reason;
}

invalid_matrix_operation
::~invalid_matrix_operation() noexcept
{
}

} } // end vital namespace
