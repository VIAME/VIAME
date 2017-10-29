#ifndef KWIVER_VITAL_KPF_BOUNCE_BUFFER_H_
#define KWIVER_VITAL_KPF_BOUNCE_BUFFER_H_

//
// The text reader is the bounce buffer between the parser's
// packet buffer and the user. At creation, these objects have
// a fixed KPF header ("g0", "id3", etc) that they know about.
//

#include <vital/kpf/vital_kpf_export.h>

#include <vital/kpf/kpf_packet.h>

#include <string>
#include <utility>

namespace kwiver {
namespace vital {
namespace kpf {

class VITAL_KPF_EXPORT packet_bounce_t
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

