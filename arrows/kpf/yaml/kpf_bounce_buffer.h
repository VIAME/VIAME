// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Bounce buffer.
 *
 *
 * This is the bounce buffer between the parser's
 * packet buffer and the user. This could probably be
 * refactored away.
 *
 */

#ifndef KWIVER_VITAL_KPF_BOUNCE_BUFFER_H_
#define KWIVER_VITAL_KPF_BOUNCE_BUFFER_H_

#include <arrows/kpf/yaml/kpf_packet.h>

#include <string>
#include <utility>

namespace kwiver {
namespace vital {
namespace kpf {

class KPF_YAML_EXPORT packet_bounce_t
{
public:
  packet_bounce_t();
  explicit packet_bounce_t( const std::string& tag );
  explicit packet_bounce_t( const packet_header_t& h );
  void init( const std::string& tag );
  void init( const packet_header_t& h );
  ~packet_bounce_t() {}

  // mutate the domain
  packet_bounce_t& set_domain( int d );

  // return this reader's packet header
  packet_header_t my_header() const;

  // transfer packet into the reader
  void set_from_buffer( const packet_t& );

  // return (true, packet) and clear the is_set flag
  // return false if set_from_buffer hasn't been called yet
  std::pair< bool, packet_t > get_packet();

protected:
  bool is_set;
  packet_header_t header;
  packet_t packet;
};

} // ...kpf
} // ...vital
} // ...kwiver

#endif

