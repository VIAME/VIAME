#ifndef KWIVER_VITAL_KPF_YAML_PARSER_H_
#define KWIVER_VITAL_KPF_YAML_PARSER_H_

#include <vital/kpf/vital_kpf_export.h>
#include <vital/kpf/kpf_parse_utils.h>
#include <vital/kpf/kpf_parser_base.h>

#include <yaml-cpp/yaml.h>

namespace kwiver {
namespace vital {
namespace kpf {

class VITAL_KPF_EXPORT kpf_yaml_parser_t: public kpf_parser_base_t
{
public:
  explicit kpf_yaml_parser_t( std::istream& is );
  ~kpf_yaml_parser_t() {}

  virtual bool get_status() const;
  virtual bool parse_next_record( packet_buffer_t& pb );

private:
  YAML::Node root;
  YAML::const_iterator current_record;
};

} // ...kpf
} // ...vital
} // ...kwiver


#endif
