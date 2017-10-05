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

#include <utility>
#include <iostream>
#include <map>
#include <sstream>

namespace kwiver {
namespace vital {
namespace kpf {

enum class VITAL_KPF_EXPORT packet_style
{
  INVALID,  // invalid, uninitialized
  ID,       // a numeric identifier (detection, track, event ID)
  TS,       // timestamp
  TSR,      // timestamp range
  LOC,      // location (2d / 3d)
  GEOM,     // bounding box
  POLY,     // polygon
  CONF,     // a confidence value
  EVENT,    // an event
  EVAL,     // an evaluation result
  ATTR,     // an attribute
  TAG,      // a tag
  KV        // a generic key/value pair
};

struct VITAL_KPF_EXPORT packet_header_t
{
  enum { NO_DOMAIN = -1 };

  packet_style style;
  int domain;
  packet_header_t(): style( packet_style::INVALID ), domain( NO_DOMAIN ) {}
  packet_header_t( packet_style s, int d ): style(s), domain(d) {}
};

VITAL_KPF_EXPORT auto packet_header_cmp = []( const packet_header_t& lhs, const packet_header_t& rhs )
{ return ( lhs.style == rhs.style )
  ? (lhs.domain < rhs.domain)
  : (lhs.style < rhs.style);
};

namespace canonical
{

struct VITAL_KPF_EXPORT bbox_t
{
  double x1, y1, x2, y2;
  bbox_t( double a, double b, double c, double d): x1(a), y1(b), x2(c), y2(d) {}
};

typedef std::size_t id_t;
typedef double timestamp_t;

struct VITAL_KPF_EXPORT timestamp_range_t
{
  timestamp_t start, stop;
};

// etc etc

} // ... canonical types

union VITAL_KPF_EXPORT payload_t
{
  payload_t(): id(0) {}
  canonical::id_t id;
  canonical::timestamp_t timestamp;
  canonical::timestamp_range_t timestamp_range;
  canonical::bbox_t bbox;
  // ... use pointers for the polygon / events
  // pay special attention to cpctor / etc...
};

struct VITAL_KPF_EXPORT packet_t
{
  packet_header_t header;
  payload_t payload;
  packet_t(): header( packet_header_t() ) {}

};


//
// text file reader:
// read a string into a buffer
// parse the buffer into a multimap of packets
// cycle through adapter interface
//
// text file writer:
// populate multimap via adapter interfaces
// dump to stream
//

class VITAL_KPF_EXPORT text_reader_t
{
public:
  text_reader_t();
  explicit text_reader_t( const std::string& tag );
  explicit text_reader_t( const packet_header_t& h );
  void init( const std::string& tag );
  void init( const packet_header_t& h );
  ~text_reader_t() {}
  void set_from_buffer( const packet_t& );
  packet_header_t my_header() const;
  std::pair< bool, packet_t > get_packet();

protected:
  bool is_set;
  packet_header_t header;
  packet_t packet;
};

typedef std::multimap< packet_header_t,
                       packet_t,
                       decltype( packet_header_cmp ) > packet_buffer_t;

class VITAL_KPF_EXPORT text_parser_t
{
public:

  explicit text_parser_t( std::istream& is );
  explicit operator bool() const;

  friend text_parser_t& operator>>( text_parser_t& t,
                                    text_reader_t& b );

  // mystery: fails to link if this is not inline?
  const packet_buffer_t& get_packet_buffer() const { return this->packet_buffer; }

private:
  bool process_reader( text_reader_t& b );
  bool parse_next_line();

  packet_buffer_t packet_buffer;
  std::istream& input_stream;
  bool reader_status;

};

VITAL_KPF_EXPORT
text_parser_t& operator>>( text_parser_t& t,
                           text_reader_t& b );

struct kpf_io_adapter_base
{
  text_reader_t text_reader;
};

template< typename USER_TYPE, typename KPF_TYPE >
struct kpf_io_adapter: public kpf_io_adapter_base
{
  // holds the two conversion functions
  USER_TYPE (*kpf2user_function)( const KPF_TYPE& );
  void (*kpf2user_inplace) ( const KPF_TYPE&, USER_TYPE& );
  KPF_TYPE (*user2kpf_function)( const USER_TYPE& );
  // what domain does this adapter read from / write to?
  int domain;
  // text reader (initialized in derived classes)

  kpf_io_adapter( USER_TYPE (*k2u)( const KPF_TYPE& ),
                  KPF_TYPE( *u2k)( const USER_TYPE& ),
                  int d ) :
    kpf2user_function( k2u ), kpf2user_inplace( nullptr ),
    user2kpf_function( u2k ), domain( d )
  {}

  kpf_io_adapter( void (*k2u)( const KPF_TYPE&, USER_TYPE& ),
                  KPF_TYPE( *u2k)( const USER_TYPE& ),
                  int d ) :
    kpf2user_function( nullptr ), kpf2user_inplace( k2u ),
    user2kpf_function( u2k ), domain( d )
  {}

  // these two methods both convert a KPF type into a user type.
  USER_TYPE operator()( const KPF_TYPE& k ) { return (user2kpf_function)( k ); }
  void operator()( const KPF_TYPE& k, USER_TYPE& u ) { (kpf2user_inplace)( k, u ); }

  // this method converts the user type and passes on the domain.
  std::pair< int, KPF_TYPE > operator()( const USER_TYPE& u )
  {
    return std::make_pair( this->domain, (kpf2user_function)( u ));
  }
};

VITAL_KPF_EXPORT
text_parser_t& operator>>( text_parser_t& t,
                           kpf_io_adapter_base& io );


template< typename USER_TYPE >
struct kpf_box_adapter: public kpf_io_adapter< USER_TYPE, canonical::bbox_t >
{
  friend std::ostream& operator<<( std::ostream& os, const kpf_box_adapter< USER_TYPE >& k );

  kpf_box_adapter( USER_TYPE (*k2u) (const canonical::bbox_t&),
                   canonical::bbox_t (*u2k)( const USER_TYPE&),
                   int d )
    : kpf_io_adapter<USER_TYPE, canonical::bbox_t>( k2u, u2k, d )
  {
    this->text_reader.init( packet_header_t( packet_style::GEOM, d ));
  }
  kpf_box_adapter( void (*k2u) (const canonical::bbox_t&, USER_TYPE& ),
                   canonical::bbox_t (*u2k)( const USER_TYPE&),
                   int d )
    : kpf_io_adapter<USER_TYPE, canonical::bbox_t>( k2u, u2k, d )
  {
    this->text_reader.init( packet_header_t( packet_style::GEOM, d ));
  }

  USER_TYPE get()
  {
    auto probe = this->text_reader.get_packet();
    // throw if ! probe->first
    // also throw if kpf2user is null, or else use a temporary?
    return (this->kpf2user_function)( probe.second.payload.bbox );
  }
  void get( USER_TYPE& u )
  {
    auto probe = this->text_reader.get_packet();
    // see above
    (this->kpf2user_inplace)( probe.second.payload.bbox, u );
  }

  canonical::bbox_t operator()( const USER_TYPE& u )
  {
    return (this->user2kpf)( u );
  }

  std::string to_str( const USER_TYPE& u ) const
  {
    canonical::bbox_t box = (this->user2kpf_function)( u );
    std::ostringstream oss;
    oss << "g" << this->domain << ": " << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y1;
    return oss.str();
  }

};

} // ...kpf
} // ...vital
} // ...kwiver

#endif
