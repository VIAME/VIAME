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

#ifndef KWIVER_VITAL_KPF_PARSE_H_
#define KWIVER_VITAL_KPF_PARSE_H_

#include <vital/kpf/vital_kpf_export.h>

#include <vital/kpf/kpf_packet.h>
#include <vital/kpf/kpf_canonical_io.h>

#include <utility>
#include <iostream>
#include <map>
#include <sstream>

namespace kwiver {
namespace vital {
namespace kpf {

//
// The text_parser reads a line from a text file and parses it into
// its packet buffer. Any syntax errors are detected during parsing.
//
// Packets are transferred out of the packet buffer via operator>>
// or the process() methods. Packets are transferred into packet_bounce_t
// objects (whch are packet-specific.)
//

//
// The packet buffer is a multimap because some packets can repeat
// (e.g. key-value packets.)
//

typedef std::multimap< packet_header_t,
                       packet_t,
                       decltype(packet_header_cmp) > packet_buffer_t;

typedef std::multimap< packet_header_t,
                       packet_t,
                       decltype(packet_header_cmp) >::const_iterator packet_buffer_cit;

class packet_bounce_t;
struct kpf_io_adapter_base;

class VITAL_KPF_EXPORT kpf_parser_base_t
{
public:
  kpf_parser_base_t() {}
  virtual ~kpf_parser_base_t() {}

  virtual bool get_status() const = 0;
  virtual bool parse_next_record( packet_buffer_t& ) = 0;
};

class VITAL_KPF_EXPORT kpf_text_parser_t: public kpf_parser_base_t
{
public:
  explicit kpf_text_parser_t( std::istream& is );
  ~kpf_text_parser_t() {}

  virtual bool get_status() const;
  virtual bool parse_next_record( packet_buffer_t& pb );

private:
  std::istream& input_stream;

};


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
  bool process( kpf_io_adapter_base& io );

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

class record_text_writer;
class packet_bounce_t;

struct VITAL_KPF_EXPORT private_endl_t
{};

class VITAL_KPF_EXPORT record_text_writer
{
public:
  explicit record_text_writer( std::ostream& os ) : s( os ) {}

  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::id_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::bbox_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::timestamp_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::kv_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::conf_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::poly_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const writer< canonical::meta_t >& io );
  friend record_text_writer& operator<<( record_text_writer& w, const private_endl_t& );

  static private_endl_t endl;

private:
  std::ostream& s;
};

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::id_t >& io );

VITAL_KPF_EXPORT
packet_bounce_t&
operator>>( packet_bounce_t& w, const writer< canonical::id_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::bbox_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::timestamp_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::kv_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::conf_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::poly_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const writer< canonical::meta_t >& io );

VITAL_KPF_EXPORT
record_text_writer&
operator<<( record_text_writer& w, const private_endl_t& e );


//
// The text reader is the bounce buffer between the parser's
// packet buffer and the user. At creation, these objects have
// a fixed KPF header ("g0", "id3", etc) that they know about.
//

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


//
// For complex types, such as bounding box, packet_bounce_t
// needs to be associated with functions to convert between
// the KPF and user types; this base class holds the
// packet_bounce_t instance.
//

struct kpf_io_adapter_base
{
  packet_bounce_t text_reader;
  kpf_io_adapter_base& set_domain( int d ) { this->text_reader.set_domain(d); return *this; }
};

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t,
                           kpf_io_adapter_base& io );


//
// The adapter class holds the infrastructure for mapping
// between user and KPF types. The user isn't intended to use
// this; instead should use classes derived from this which
// are specialized on KPF_TYPE.
//

template< typename USER_TYPE, typename KPF_TYPE >
struct kpf_io_adapter: public kpf_io_adapter_base
{
  // holds the two conversion functions
  USER_TYPE (*kpf2user_function)( const KPF_TYPE& );
  void (*kpf2user_inplace) ( const KPF_TYPE&, USER_TYPE& );
  KPF_TYPE (*user2kpf_function)( const USER_TYPE& );

  // text reader (initialized in derived classes)

  kpf_io_adapter( USER_TYPE (*k2u)( const KPF_TYPE& ),
                  KPF_TYPE( *u2k)( const USER_TYPE& ) ) :
    kpf2user_function( k2u ), kpf2user_inplace( nullptr ), user2kpf_function( u2k )
  {}

  kpf_io_adapter( void (*k2u)( const KPF_TYPE&, USER_TYPE& ),
                  KPF_TYPE( *u2k)( const USER_TYPE& ) ) :
    kpf2user_function( nullptr ), kpf2user_inplace( k2u ), user2kpf_function( u2k )
  {}

  // these two methods both convert a KPF type into a user type.
  USER_TYPE operator()( const KPF_TYPE& k ) { return (user2kpf_function)( k ); }
  void operator()( const KPF_TYPE& k, USER_TYPE& u ) { (kpf2user_inplace)( k, u ); }

  // this method converts the user type and passes on the domain.
  std::pair< int, KPF_TYPE > operator()( const USER_TYPE& u, int domain )
  {
    return std::make_pair( domain, (kpf2user_function)( u ));
  }
};


//
// This is a KPF I/O adapter for bounding boxes.
//

template< typename USER_TYPE >
struct kpf_box_adapter: public kpf_io_adapter< USER_TYPE, canonical::bbox_t >
{

  kpf_box_adapter( USER_TYPE (*k2u) (const canonical::bbox_t&),
                   canonical::bbox_t (*u2k)( const USER_TYPE&) )
    : kpf_io_adapter<USER_TYPE, canonical::bbox_t>( k2u, u2k )
  {
    this->text_reader.init( packet_header_t( packet_style::GEOM ));
  }
  kpf_box_adapter( void (*k2u) (const canonical::bbox_t&, USER_TYPE& ),
                   canonical::bbox_t (*u2k)( const USER_TYPE&) )
    : kpf_io_adapter<USER_TYPE, canonical::bbox_t>( k2u, u2k )
  {
    this->text_reader.init( packet_header_t( packet_style::GEOM ));
  }

  USER_TYPE get()
  {
    auto probe = this->text_reader.get_packet();
    // throw if ! probe->first
    // also throw if kpf2user is null, or else use a temporary?
    return (this->kpf2user_function)( probe.second.bbox );
  }
  void get( USER_TYPE& u )
  {
    auto probe = this->text_reader.get_packet();
    // see above
    (this->kpf2user_inplace)( probe.second.bbox, u );
  }

  void get( kpf_reader_t& parser, USER_TYPE& u )
  {
    // same throwing issues
    parser.process( *this );
    this->get( u );
  }

  canonical::bbox_t operator()( const USER_TYPE& u )
  {
    return this->user2kpf_function( u );
  }

};

//
// This is a KPF I/O adapter for polygons.
//

template< typename USER_TYPE >
struct kpf_poly_adapter: public kpf_io_adapter< USER_TYPE, canonical::poly_t >
{

  kpf_poly_adapter( USER_TYPE (*k2u) (const canonical::poly_t&),
                    canonical::poly_t (*u2k)( const USER_TYPE&) )
    : kpf_io_adapter<USER_TYPE, canonical::poly_t>( k2u, u2k )
  {
    this->text_reader.init( packet_header_t( packet_style::POLY ));
  }
  kpf_poly_adapter( void (*k2u) (const canonical::poly_t&, USER_TYPE& ),
                    canonical::poly_t (*u2k)( const USER_TYPE&) )
    : kpf_io_adapter<USER_TYPE, canonical::poly_t>( k2u, u2k )
  {
    this->text_reader.init( packet_header_t( packet_style::POLY ));
  }

  USER_TYPE get()
  {
    auto probe = this->text_reader.get_packet();
    // throw if ! probe->first
    // also throw if kpf2user is null, or else use a temporary?
    return (this->kpf2user_function)( probe.second.poly );
  }
  void get( USER_TYPE& u )
  {
    auto probe = this->text_reader.get_packet();
    // see above
    (this->kpf2user_inplace)( probe.second.poly, u );
  }

  void get( kpf_reader_t& parser, USER_TYPE& u )
  {
    // same throwing issues
    parser.process( *this );
    this->get( u );
  }

  canonical::poly_t operator()( const USER_TYPE& u )
  {
    return this->user2kpf_function( u );
  }

};


VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::bbox_t >& r );

VITAL_KPF_EXPORT
kpf_reader_t& operator>>( kpf_reader_t& t, const reader< canonical::poly_t >& r );

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

} // ...kpf
} // ...vital
} // ...kwiver

#endif
