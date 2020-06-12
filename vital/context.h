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
