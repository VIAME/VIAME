// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_KWIVER_LOGGER_MANAGER_H_
#define KWIVER_KWIVER_LOGGER_MANAGER_H_

#include "kwiver_logger.h"
#include <vital/logger/vital_logger_export.h>

#include <string>
#include <memory>
#include <vital/noncopyable.h>

namespace kwiver {
namespace vital {
namespace logger_ns {

  class kwiver_logger_factory;

}

// ----------------------------------------------------------------
/** Logger manager.
 *
 * This class represents the main top level logic for the KWIVER
 * logger. Only one object of this type is required, so this is a
 * singleton created by the static instance() method.
 */
class VITAL_LOGGER_EXPORT kwiver_logger_manager
  :private kwiver::vital::noncopyable
{
public:
  ~kwiver_logger_manager();

  /** Get the single instance of this class. */
  static kwiver_logger_manager * instance();

  /**
   * @brief Get name of current logger factory.
   *
   * This method returns the name of the currently active logger
   * factory.
   *
   * @return Name of logger factory.
   */
  std::string const&  get_factory_name() const;

  /**
   * @brief Establish a new logger factory.
   *
   * The specified logger factory object is installed as the current
   * factory and the old factory is returned. This is useful for
   * setting up loggers that are tightly coupled with the application.
   *
   * @param fact Pointer to new factory.
   */
  void set_logger_factory( std::unique_ptr< logger_ns::kwiver_logger_factory >&& fact );

private:
  friend VITAL_LOGGER_EXPORT logger_handle_t
    get_logger( const char * const name );

  kwiver_logger_manager();
  void load_factory( std::string const& lib_name );

  class impl;
  const std::unique_ptr< impl > m_impl;

  static kwiver_logger_manager * s_instance;
}; // end class kwiver_logger_manager

} } // end namespace kwiver

#endif
