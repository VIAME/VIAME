// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief algorithm exception implementations
 */

#include "algorithm.h"
#include <sstream>

namespace kwiver {
namespace vital {

algorithm_exception
::algorithm_exception(std::string type,
                      std::string impl,
                      std::string reason) noexcept
  : m_algo_type(type)
  , m_algo_impl(impl)
  , m_reason(reason)
{
  // generic what string
  std::ostringstream sstr;
  sstr << "[algo::" << type << "::" << impl << "]: "
       << reason;
  m_what = sstr.str();
}

algorithm_exception
::~algorithm_exception() noexcept
{
}

algorithm_configuration_exception
::algorithm_configuration_exception(std::string type,
                                    std::string impl,
                                    std::string reason) noexcept
  : algorithm_exception(type, impl, reason)
{
  std::ostringstream sstr;
  sstr << "Failed to configure algorithm "
       << "\"" << m_algo_type << "::" << m_algo_impl << "\" due to: "
       << reason;
  m_what = sstr.str();
}

algorithm_configuration_exception
::~algorithm_configuration_exception() noexcept
{
}

invalid_name_exception
::invalid_name_exception(std::string type,
                         std::string impl) noexcept
  : algorithm_exception(type, impl, "")
{
  std::ostringstream sstr;
  sstr << "Invalid algorithm impl name \"" << m_algo_impl << "\""
       << "for type \"" << m_algo_type << "\".";
  m_what = sstr.str();
}

invalid_name_exception
::~invalid_name_exception() noexcept
{
}

} } // end namespace vital
