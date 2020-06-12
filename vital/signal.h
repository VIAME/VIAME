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

/**
 * \file
 * \brief Interface for signal class
 */

#ifndef VITAL_SIGNAL_H_
#define VITAL_SIGNAL_H_

#include <vital/context.h>

#include <functional>
#include <unordered_map>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/**
 * \brief Base class for ::signal.
 *
 * This class encapsulates some functionality used by ::signal which must not
 * be templated. This class is an implemenation detail and should not be used
 * directly.
 */
class signal_base
{
protected:
  friend class context;

  inline signal_base() = default;
  inline ~signal_base() = default;

  signal_base( signal_base const& ) = delete;
  signal_base& operator=( signal_base const& ) = delete;

  virtual void disconnect( context* ctx ) = 0;

  std::mutex m_mutex;
};

// ----------------------------------------------------------------------------
/**
 * \brief Encapsulation class for an event signal.
 *
 * This class represents a signal which may be emitted when an event occurs.
 * Signals provide a mechanism by which interested listeners may connect to
 * the signal in order to receive notification of the event. Because connected
 * slots are executed synchronously, it is generally recommended such slots
 * execute as quickly as possible.
 *
 * The thread on which connected slots will be invoked is specified by the
 * logic which emits the signal, and in general should be considered as
 * unspecified. The order in which connected slots are executed is also
 * unspecified and may change between program executions or when the set of
 * connected slots changes.
 *
 * Concurrent emission of a signal from multiple threads is not currently
 * supported.
 *
 * Currently, slots must have an associated ::context. This is used to ensure
 * that the slot is not executed after its required resources are destroyed.
 * Users should ensure that the associated ::context is destroyed before any
 * resources required by the slot are destroyed.
 */
template < typename... Args >
class signal : protected signal_base
{
public:
  using slot_t = std::function< void ( Args... ) >;

  signal() = default;
  ~signal()
  {
    std::lock_guard< std::mutex > lock{ this->m_mutex };
    for ( auto const& iter : m_slots )
    {
      iter.first->disconnect( this );
    }
  }

  /**
   * \brief Emit the signal.
   *
   * This method triggers emission of this signal. The supplied arguments are
   * passed to any connected slots.
   */
  void
  operator()( Args... args )
  {
    std::lock_guard< std::mutex > lock{ this->m_mutex };
    for ( auto const& iter : m_slots )
    {
      iter.second( args... );
    }
  }

  /**
   * \brief Connect a slot to this signal.
   *
   * This method connects the specified \p slot, which is owned by the
   * specified ::context \p ctx, to this signal. The slot will be called when
   * the signal is emitted, and will be disconnected when its associated
   * context is destroyed.
   *
   * \warning
   * Destruction of the ::context during the execution of this method will
   * result in undefined behavior and may cause the program to deadlock or
   * crash.
   */
  void
  connect( context* ctx, slot_t&& slot )
  {
    std::lock_guard< std::mutex > lock{ this->m_mutex };
    this->m_slots.emplace( ctx, std::move( slot ) );
    ctx->connect( this );
  }

protected:
  void
  disconnect( context* ctx ) final
  {
    // WARNING: This must be called with m_mutex already locked
    m_slots.erase( ctx );
  }

private:
  std::unordered_map< context*, slot_t > m_slots;
};

} // namespace vital

} // namespace kwiver

#endif
