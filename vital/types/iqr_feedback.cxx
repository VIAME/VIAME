// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
vital::uid
iqr_feedback
::query_id() const
{
  return m_query_id;
}

// ----------------------------------------------------------------------------
void
iqr_feedback
::set_query_id( vital::uid const& id )
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
