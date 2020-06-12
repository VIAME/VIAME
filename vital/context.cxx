/*ckwg +30
 * Copyright 2020 by Kitware, Inc.
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
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

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
