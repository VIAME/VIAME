// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// Map-based implementations of the metadata_stream interfaces.

#include <viame/core_types/metadata_stream_from_map.h>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
metadata_istream_from_map
::metadata_istream_from_map( map_t const& map )
  : m_map{ &map },
    m_it{ map.cbegin() }
{}

// ----------------------------------------------------------------------------
metadata_istream_from_map
::~metadata_istream_from_map()
{}

// ----------------------------------------------------------------------------
metadata_istream_from_map::map_t const&
metadata_istream_from_map
::map() const
{
  return *m_map;
}

// ----------------------------------------------------------------------------
metadata_istream_from_map::iterator_t
metadata_istream_from_map
::iterator() const
{
  return m_it;
}

// ----------------------------------------------------------------------------
frame_id_t
metadata_istream_from_map
::frame_number() const
{
  if( at_end() )
  {
    throw std::invalid_argument(
      "metadata_istream_from_map::frame_number() "
      "called at end of stream" );
  }
  return m_it->first;
}

// ----------------------------------------------------------------------------
metadata_vector
metadata_istream_from_map
::metadata()
{
  if( at_end() )
  {
    throw std::invalid_argument(
      "metadata_istream_from_map::metadata() "
      "called at end of stream" );
  }
  return m_it->second;
}

// ----------------------------------------------------------------------------
bool
metadata_istream_from_map
::next_frame()
{
  if( at_end() )
  {
    return false;
  }

  ++m_it;

  return !at_end();
}

// ----------------------------------------------------------------------------
bool
metadata_istream_from_map
::at_end() const
{
  return m_it == m_map->cend();
}

// ----------------------------------------------------------------------------
metadata_ostream_from_map
::metadata_ostream_from_map( map_t& map )
  : m_map{ &map },
    m_at_end{ false }
{}

// ----------------------------------------------------------------------------
metadata_ostream_from_map
::~metadata_ostream_from_map()
{}

// ----------------------------------------------------------------------------
metadata_ostream_from_map::map_t&
metadata_ostream_from_map
::map() const
{
  return *m_map;
}

// ----------------------------------------------------------------------------
bool
metadata_ostream_from_map
::write_frame( frame_id_t frame_number, metadata_vector const& metadata )
{
  if( at_end() )
  {
    throw std::invalid_argument(
      "metadata_ostream_from_map::write_frame() "
      "called at end of stream" );
  }

  auto const it = m_map->find( frame_number );
  if( it == m_map->end() )
  {
    m_map->emplace( frame_number, metadata );
  }
  else
  {
    it->second.insert( it->second.end(), metadata.begin(), metadata.end() );
  }

  return true;
}

// ----------------------------------------------------------------------------
void
metadata_ostream_from_map
::write_end()
{
  m_at_end = true;
}

// ----------------------------------------------------------------------------
bool
metadata_ostream_from_map
::at_end() const
{
  return m_at_end;
}

} // namespace vital

} // namespace kwiver
