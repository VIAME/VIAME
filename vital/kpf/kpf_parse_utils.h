#ifndef KWIVER_VITAL_KPR_PARSE_UTILS_H_
#define KWIVER_VITAL_KPR_PARSE_UTILS_H_

#include <vital/kpf/vital_kpf_export.h>

#include <vital/kpf/kpf_packet.h>

#include <string>
#include <vector>
#include <tuple>
#include <map>

namespace kwiver {
namespace vital {
namespace kpf {

//
// The packet buffer is a multimap because some packets can repeat
// (e.g. key-value packets.)
//

typedef std::multimap< packet_header_t,
                       packet_t,
                       packet_header_cmp > packet_buffer_t;

typedef std::multimap< packet_header_t,
                       packet_t,
                       packet_header_cmp >::const_iterator packet_buffer_cit;


typedef std::tuple< bool, std::string, int > header_parse_t;

bool VITAL_KPF_EXPORT packet_header_parser( const std::string& s,
                      packet_header_t& packet_header,
                      bool expect_colon );

bool VITAL_KPF_EXPORT
packet_parser( const std::vector< std::string >& tokens,
               packet_buffer_t& packet_buffer );

header_parse_t VITAL_KPF_EXPORT parse_header( const std::string& s, bool expect_colon );

std::pair< bool, size_t > VITAL_KPF_EXPORT packet_payload_parser (
  size_t index,
  const std::vector< std::string >& tokens,
  packet_t& packet );


} // ...kpf
} // ...vital
} // ...kwiver

#endif
