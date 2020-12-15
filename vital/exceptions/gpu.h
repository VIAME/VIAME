// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for GPU exceptions
 */

#ifndef VITAL_CORE_EXCEPTIONS_GPU_H
#define VITAL_CORE_EXCEPTIONS_GPU_H

#include <string>

#include <vital/exceptions/base.h>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Generic GPU exception
class VITAL_EXCEPTIONS_EXPORT gpu_exception
  : public vital_exception
{
public:
  /// Constructor
  gpu_exception() noexcept;

  /// Destructor
  virtual ~gpu_exception() noexcept;
};

// ------------------------------------------------------------------
/// Video runtime error.
/*
 * This exception is thrown when the GPU is unable to allocate memory
 */
class VITAL_EXCEPTIONS_EXPORT gpu_memory_exception
  : public gpu_exception
{
public:
  /// Constructor
  gpu_memory_exception( std::string const& msg ) noexcept;

  /// Destructor
  virtual ~gpu_memory_exception() noexcept;
};

} } // end namespace

#endif /* VITAL_CORE_EXCEPTIONS_GPU_H */
