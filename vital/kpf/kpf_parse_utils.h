#ifndef KWIVER_VITAL_KPR_PARSING_UTILS_H_
#define KWIVER_VITAL_KPR_PARSING_UTILS_H_

#include <vital/kpf/vital_kpf_export.h>

#include <vital/kpf/kpf_parse.h>

#include <tuple>
#include <string>

namespace kwiver {
namespace vital {
namespace kpf {

typedef std::tuple< bool, std::string, int > header_parse_t;

header_parse_t VITAL_KPF_EXPORT parse_header( const std::string& s );


} // ...kpf
} // ...vital
} // ...kwiver




#endif
