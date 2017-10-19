#include "kpf_packet.h"

#include <stdexcept>
#include <sstream>

namespace kwiver {
namespace vital {
namespace kpf {

packet_t
::~packet_t()
{
  // all payload variants trivially destructable so far
}

packet_t
::packet_t( const packet_t& other ):
  header( other.header )
{
  *this = other;
}

packet_t&
packet_t
::operator=( const packet_t& other )
{
  this->header = other.header;
  switch (this->header.style)
  {
  case packet_style::INVALID: break;
  case packet_style::ID:      this->id = other.id; break;
  case packet_style::TS:      this->timestamp = other.timestamp; break;
  case packet_style::TSR:     this->timestamp_range = other.timestamp_range; break;
  case packet_style::KV:      this->kv = other.kv; break;
  case packet_style::CONF:    this->conf = other.conf; break;
  case packet_style::GEOM:    this->bbox = other.bbox; break;
  default:
    {
      std::ostringstream oss;
      oss << "Unhandled cpctor for style " << static_cast<int>(this->header.style) << " (domain " << this->header.domain << ")";
      throw std::logic_error( oss.str() );
    }
  }
  return *this;
}

} // ...kpf
} // ...vital
} // ...kwiver

