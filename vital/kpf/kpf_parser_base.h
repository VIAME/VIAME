#ifndef KWIVER_VITAL_KPF_PARSER_BASE_
#define KWIVER_VITAL_KPF_PARSER_BASE_

#include <vital/kpf/kpf_parse_utils.h>

namespace kwiver {
namespace vital {
namespace kpf {



class kpf_parser_base_t
{
public:
  kpf_parser_base_t() {}
  virtual ~kpf_parser_base_t() {}

  virtual bool get_status() const = 0;
  virtual bool parse_next_record( packet_buffer_t& ) = 0;
};


} // ...kpf
} // ...vital
} // ...kwiver

#endif
