// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <vital/context.h>
#include <vital/signal.h>

#include <thread>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
context
::context()
{
}

// ----------------------------------------------------------------------------
context::~context()
{
  // Disconnect from all connected signals
  std::unique_lock< std::mutex > lock{ this->m_mutex };
  while ( !this->m_connections.empty() )
  {
    auto failed = false;
    auto iter = this->m_connections.begin();
    while ( iter != this->m_connections.end() )
    {
      // Attempt to disconnect; if the signal is being emitted or destroyed, we
      // will not be able to get the mutex, so skip it and try again after
      // dropping our own lock to allow the signal to finish what it's doing
      auto* const connection = *iter;
      std::unique_lock< std::mutex > connection_lock{
        connection->m_mutex, std::try_to_lock };
      if ( connection_lock.owns_lock() )
      {
        connection->disconnect( this );
        iter = this->m_connections.erase( iter );
      }
      else
      {
        failed = true;
        ++iter;
      }
    }
    if ( failed )
    {
      // Need to back off our lock to let a connected signal do something
      lock.unlock();
      std::this_thread::yield();
      lock.lock();
    }
  }
}

// ----------------------------------------------------------------------------
void
context
::connect( signal_base* signal )
{
  std::lock_guard< std::mutex > lock{ this->m_mutex };
  this->m_connections.emplace( signal );
}

// ----------------------------------------------------------------------------
void
context
::disconnect( signal_base* signal )
{
  std::lock_guard< std::mutex > lock{ this->m_mutex };
  this->m_connections.erase( signal );
}

} // namespace vital

} // namespace kwiver
