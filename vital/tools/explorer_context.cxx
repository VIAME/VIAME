/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "explorer_plugin.h"
#include "explorer_context_priv.h"

namespace kwiver {
namespace vital {

// ==================================================================
// --- Explorer context methods ---
// ------------------------------------------------------------------
explorer_context::
explorer_context( explorer_context::priv* pp )
  : p( pp )
{ }


explorer_context::
~explorer_context()
{ }


// ------------------------------------------------------------------
std::ostream&
explorer_context::
output_stream() const
{
  return *p->m_out_stream;
}


// ------------------------------------------------------------------
kwiversys::CommandLineArguments*
explorer_context::
command_line_args()
{
  return &p->m_args;
}


// ------------------------------------------------------------------
const std::string&
explorer_context::
formatting_type() const
{
  return p->formatting_type;
}


// ------------------------------------------------------------------
std::string
explorer_context::
wrap_text( const std::string& text ) const
{
  return p->m_wtb.wrap_text( text );
}

// ------------------------------------------------------------------
//
// display full factory list
//
void
explorer_context::
display_attr( const kwiver::vital::plugin_factory_handle_t fact ) const
{
  p->display_attr( fact );
}

// ------------------------------------------------------------------
bool explorer_context::if_detail() const { return p->opt_detail; }
bool explorer_context::if_brief() const { return p->opt_brief; }

} } // end namespace
