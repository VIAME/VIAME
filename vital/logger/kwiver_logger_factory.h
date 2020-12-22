// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_KWIVER_LOGGER_FACTORY_H_
#define KWIVER_KWIVER_LOGGER_FACTORY_H_

#include "kwiver_logger.h"
#include <vital/noncopyable.h>

namespace kwiver {
namespace vital {
namespace logger_ns {

// ----------------------------------------------------------------
/** Factory for underlying logger.
 *
 * This class is the abstract base class that adapts the KWIVER logger
 * to the underlying logging system.
 *
 * An object of this type can be created early in the program
 * execution (i.e. static initializer time), which is before the
 * initialize method is called.
 */
class VITAL_LOGGER_EXPORT kwiver_logger_factory
  : private kwiver::vital::noncopyable
{
public:
  /**
   * @brief Create logger factory
   *
   * The name for this factory should describe the logger type that is
   * being created.
   *
   * Call get_name() to access this name.
   *
   * @param name  Name of this logger factory
   */
  kwiver_logger_factory( std::string const& name );
  virtual ~kwiver_logger_factory();

  /**
   * @brief Get pointer to named logger.
   *
   * This method returns a pointer to a named logger. The underlying
   * log provider determines what is actually done here.
   *
   * @param name Name of logger object.
   *
   * @return
   */
  virtual logger_handle_t get_logger( std::string const& name ) = 0;

  /**
   * @brief Get logger factory name.
   *
   * Returns the name associated with this logger factory.
   *
   * @return Name or type of logger created by this factory.
   */
  std::string const& get_factory_name() const;

private:
  std::string m_name; // factory name
}; // end class kwiver_logger_factory

} } } // end namespace

#endif
