// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for context class
 */

#ifndef VITAL_CONTEXT_H_
#define VITAL_CONTEXT_H_

#include <vital/vital_export.h>

#include <memory>
#include <mutex>
#include <unordered_set>

namespace kwiver {

namespace vital {

class signal_base;

template < typename... Args > class signal;

// ----------------------------------------------------------------------------
/**
 * \brief Slot execution context.
 *
 * This class represents a context for a slot connected to a ::signal. The
 * context is used to manage resources that may be required by slots. When the
 * context is destroyed, any connections associated with the context will be
 * disconnected.
 *
 * \warning
 * Destroying a context from within a connected slot is a logic error and will
 * likely cause the program to deadlock or exhibit undefined behavior.
 */
class VITAL_EXPORT context
{
public:
  context();
  ~context();

private:
  template < typename... Args > friend class signal;

  void connect( signal_base* signal );
  void disconnect( signal_base* signal );

  std::mutex m_mutex;
  std::unordered_set< signal_base* > m_connections;
};

} // namespace vital

} // namespace kwiver

#endif
