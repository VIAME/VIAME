/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "edge.h"
#include "edge_exception.h"

namespace vistk
{

edge
::~edge()
{
}

bool
edge
::makes_dependency() const
{
  return true;
}

bool
edge
::has_data() const
{
  return (datum_count() != 0);
}

edge_datum_t
edge
::get_datum()
{
  edge_datum_t const dat = peek_datum();

  pop_datum();

  return dat;
}

void
edge
::set_required_by_downstream(bool required)
{
  m_required = required;
}

bool
edge
::required_by_downstream() const
{
  return m_required;
}

void
edge
::set_upstream_process(process_t process)
{
  if (!process)
  {
    throw null_process_connection();
  }

  if (m_upstream)
  {
    throw input_already_connected(m_upstream->name(), process->name());
  }

  m_upstream = process;
}

void
edge
::set_downstream_process(process_t process)
{
  if (!process)
  {
    throw null_process_connection();
  }

  if (m_downstream)
  {
    throw output_already_connected(m_downstream->name(), process->name());
  }

  m_downstream = process;
}

edge
::edge(config_t const& /*config*/)
  : m_required(true)
{
}

} // end namespace vistk
