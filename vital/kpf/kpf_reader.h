//
// for text files (at least), each line is a record of packets.
//
// I read the file. I get:
//
// - header (vector of meta packets)
// - content (vector of records)
// - content-descriptor (map of packet -> count)
//
// A record is:
// - a std::map< packet_t, std::unique_ptr<packet_payload_t> >
// ... one instance of packet_t is 'key', so a record may have multiple key/value pairs.
//
// ... packet_t is a pair <packet_enum_t, domain_t>
// ... packet_payload_t is an ABC, static_cast based on packet_enum_t
//
// record_writer rw;
// rw_index_t geo = rw.add( kpf::geometry, kpf::geometry::PIXEL, []( vgl_box_2d& b ){ return make_tuple( b.lx, ... ); } );
// ...
// hmm need to do that because we need to preallocate the domains?
//
// rw.write( geo, box_list[1] );
// ...no, what's the signature there?
//
// howabout:
//
// auto cvt_box = []( const vgl_box_2d<double>& b ) { return kpf::geo::Type( b.lx, ... );
// record_writer rw;
// rw_tag_t geo_tag = rw.add( kpf::geo::Tag, kpf::geometry::PIXEL ); // now have the tag and the domain
// rw_tag_t frame_tag = rw.add( kpf::id::Tag, kpf::next, "detection ID" ); // added to meta; next is a different type to force overload
// rw_tag_t track_tag = rw.add( kpf::id::Tag, kpf::next, "track ID" ); // added to meta
// ...
// rw << frame_tag << frame_id << track_tag << track_id << geo_tag << cvt_box( b1 ) << rw::endl;
// ...
// ...and the op<< for rw has the state to check the decltype for the tags for the following call?
// ... hmmm!
//
// Reading?
//
// ... see tmp.cxx
//
//

#ifndef KWIVER_VITAL_KPF_READER_H_
#define KWIVER_VITAL_KPF_READER_H_


#include <vital/kpf/vital_kpf_export.h>

#include <vital/kpf/kpf_packet.h>
#include <vital/kpf/kpf_canonical_io.h>
#include <vital/kpf/kpf_parse_utils.h>
#include <vital/kpf/kpf_parser_base.h>
#include <vital/kpf/kpf_bounce_buffer.h>
#include <vital/kpf/kpf_canonical_io_adapter_base.h>

#include <utility>
#include <iostream>
#include <map>
#include <sstream>

namespace kwiver {
namespace vital {
namespace kpf {

class VITAL_KPF_EXPORT kpf_reader_t
{
public:

  explicit kpf_reader_t( kpf_parser_base_t& parser );
  explicit operator bool() const;

  // load more packets, if necessary
  bool next();

  // push packets into the text_reader
  friend kpf_reader_t& operator>>( kpf_reader_t& t,
                                   packet_bounce_t& b );

  // pull packets into the text_reader
  bool process( packet_bounce_t& b );
  bool process( kpf_canonical_io_adapter_base& io );

  // mystery: fails to link if this is not inline?
  const packet_buffer_t& get_packet_buffer() const { return this->packet_buffer; }

  // clear the packet buffer
  void flush() { this->packet_buffer.clear(); this->meta_buffer.clear(); }

  // look for a packet matching the header; if found,
  // return true, remove from buffer, return the packet
  // if not found, return false
  std::pair< bool, packet_t > transfer_packet_from_buffer( const packet_header_t& h );

  // like above, but specifically for kv (key/value) packets with a
  // particular key
  std::pair< bool, packet_t > transfer_kv_packet_from_buffer( const std::string& key );

  // return any meta packets
  std::vector< std::string > get_meta_packets() const;

private:
  bool process_reader( packet_bounce_t& b );
  bool parse_next_line();
  bool verify_reader_status();

  packet_buffer_t packet_buffer;
  std::vector< std::string > meta_buffer;
  bool reader_status;

  kpf_parser_base_t& parser;
};

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t,
                          packet_bounce_t& b );
//
//
//


VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::bbox_t >& r );

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::poly_t >& r );

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::activity_t >& r );

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::id_t >& r );

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::timestamp_t >& r );

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::kv_t >& r );

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::conf_t >& r );

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::meta_t >& r );

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::timestamp_range_t >& r );



} // ...kpf
} // ...vital
} // ...kwiver

#endif
