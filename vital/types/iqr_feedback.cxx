/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

/**
 * \file
 * \brief This file contains the implementation of iqr feedback
 */

#include "iqr_feedback.h"

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------------------
iqr_feedback
::iqr_feedback()
{
}

// ----------------------------------------------------------------------------
uid
iqr_feedback
::query_id() const
{
  return m_query_id;
}

// ----------------------------------------------------------------------------
void
iqr_feedback
::set_query_id( uid const& id )
{
  m_query_id = id;
}

// ----------------------------------------------------------------------------
std::vector< unsigned > const&
iqr_feedback
::positive_ids() const
{
  return m_positive_ids;
}

// ----------------------------------------------------------------------------
void
iqr_feedback
::set_positive_ids( std::vector< unsigned > const& ids )
{
  m_positive_ids = ids;
}

// ----------------------------------------------------------------------------
std::vector< unsigned > const&
iqr_feedback
::negative_ids() const
{
  return m_negative_ids;
}

// ----------------------------------------------------------------------------
void
iqr_feedback
::set_negative_ids( std::vector< unsigned > const& ids )
{
  m_negative_ids = ids;
}

} } // end namespace
