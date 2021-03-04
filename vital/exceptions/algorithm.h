// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief algorithm exceptions interfaces
 */

#ifndef VITAL_CORE_EXCEPTIONS_ALGORITHM_H
#define VITAL_CORE_EXCEPTIONS_ALGORITHM_H

#include "base.h"
#include <string>

namespace kwiver {
namespace vital {

// ------------------------------------------------------------------
/// Base class for all algorithm related exceptions
/**
 * \ingroup exceptions
 */
class VITAL_EXCEPTIONS_EXPORT algorithm_exception
  : public vital_exception
{
  public:
    /// Constructor
    algorithm_exception(std::string type,
                        std::string impl,
                        std::string reason) noexcept;
    /// Deconstructor
    virtual ~algorithm_exception() noexcept;

    /// The name of the algorithm type
    std::string m_algo_type;

    /// The name of the algorithm implementation
    std::string m_algo_impl;

    /// String explanation of the reason for the exception
    std::string m_reason;
};

// ------------------------------------------------------------------
/// Exception for when an algorithm receives an invalid configuration
/**
 * \ingroup exceptions
 */
class VITAL_EXCEPTIONS_EXPORT algorithm_configuration_exception
  : public algorithm_exception
{
  public:
    /// Constructor
    algorithm_configuration_exception(std::string type,
                                      std::string impl,
                                      std::string reason) noexcept;
    /// Destructor
    virtual ~algorithm_configuration_exception() noexcept;
};

// ------------------------------------------------------------------
/// Exception for when checking an invalid impl name against an algo def
/**
 * \ingroup exceptions
 */
class VITAL_EXCEPTIONS_EXPORT invalid_name_exception
  : public algorithm_exception
{
  public:
    /// Constructor
    invalid_name_exception(std::string type,
                           std::string impl) noexcept;

    /// Destructor
    virtual ~invalid_name_exception() noexcept;
};

} } // end namespace vital

#endif // VITAL_CORE_EXCEPTIONS_ALGORITHM_H
