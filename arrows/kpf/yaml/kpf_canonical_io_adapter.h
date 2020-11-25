// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Base class for adapters for complex types.
 *
 * For complex types (bounding boxes, activities), there's probably not
 * a one-to-one mapping between the canonical KPF representation and the user's
 * data structure.
 *
 * This holds the (templated) canonical_io_adapter_base as well as some
 * derived classes for enabling user-types to map more easily into:
 *
 * -- bounding boxes
 * -- polygons
 * -- activities
 *
 */

#ifndef KWIVER_VITAL_KPF_CANONICAL_IO_ADAPTER_H_
#define KWIVER_VITAL_KPF_CANONICAL_IO_ADAPTER_H_

#include <arrows/kpf/yaml/kpf_reader.h>
#include <arrows/kpf/yaml/kpf_canonical_io_adapter_base.h>

namespace kwiver {
namespace vital {
namespace kpf {

/**
 * \brief Base class for kpf <-> user translation
 *
 * The adapter base class holds the infrastructure for mapping
 * between user and KPF types. The user isn't intended to use
 * this; instead should use classes derived from this which
 * are specialized on KPF_TYPE.
 *
 * The user provides two functions; one constructs a
 * user type from the canonical type, the other constructs a canonical
 * type from the user's type.
 */

template< typename USER_TYPE, typename KPF_TYPE >
struct kpf_io_adapter: public kpf_canonical_io_adapter_base
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

/**
 * \brief Bounding box adapter.
 *
 */

template< typename USER_TYPE >
struct kpf_box_adapter: public kpf_io_adapter< USER_TYPE, canonical::bbox_t >
{

  kpf_box_adapter( USER_TYPE (*k2u) (const canonical::bbox_t&),
                   canonical::bbox_t (*u2k)( const USER_TYPE&) )
    : kpf_io_adapter<USER_TYPE, canonical::bbox_t>( k2u, u2k )
  {
    this->packet_bounce.init( packet_header_t( packet_style::GEOM ));
  }
  kpf_box_adapter( void (*k2u) (const canonical::bbox_t&, USER_TYPE& ),
                   canonical::bbox_t (*u2k)( const USER_TYPE&) )
    : kpf_io_adapter<USER_TYPE, canonical::bbox_t>( k2u, u2k )
  {
    this->packet_bounce.init( packet_header_t( packet_style::GEOM ));
  }

  USER_TYPE get()
  {
    auto probe = this->packet_bounce.get_packet();
    // throw if ! probe->first
    // also throw if kpf2user is null, or else use a temporary?
    return (this->kpf2user_function)( probe.second.bbox );
  }
  void get( USER_TYPE& u )
  {
    auto probe = this->packet_bounce.get_packet();
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

/**
 * \brief Polygon adapter.
 *
 */

template< typename USER_TYPE >
struct kpf_poly_adapter: public kpf_io_adapter< USER_TYPE, canonical::poly_t >
{

  kpf_poly_adapter( USER_TYPE (*k2u) (const canonical::poly_t&),
                    canonical::poly_t (*u2k)( const USER_TYPE&) )
    : kpf_io_adapter<USER_TYPE, canonical::poly_t>( k2u, u2k )
  {
    this->packet_bounce.init( packet_header_t( packet_style::POLY ));
  }
  kpf_poly_adapter( void (*k2u) (const canonical::poly_t&, USER_TYPE& ),
                    canonical::poly_t (*u2k)( const USER_TYPE&) )
    : kpf_io_adapter<USER_TYPE, canonical::poly_t>( k2u, u2k )
  {
    this->packet_bounce.init( packet_header_t( packet_style::POLY ));
  }

  USER_TYPE get()
  {
    auto probe = this->packet_bounce.get_packet();
    // throw if ! probe->first
    // also throw if kpf2user is null, or else use a temporary?
    return (this->kpf2user_function)( probe.second.poly );
  }
  void get( USER_TYPE& u )
  {
    auto probe = this->packet_bounce.get_packet();
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

/**
 * \brief Activity adapter.
 *
 */

template< typename USER_TYPE >
struct kpf_act_adapter: public kpf_io_adapter< USER_TYPE, canonical::activity_t >
{

  kpf_act_adapter( USER_TYPE (*k2u) (const canonical::activity_t&),
                   canonical::activity_t (*u2k)( const USER_TYPE&) )
    : kpf_io_adapter<USER_TYPE, canonical::activity_t>( k2u, u2k )
  {
    this->packet_bounce.init( packet_header_t( packet_style::ACT ));
  }
  kpf_act_adapter( void (*k2u) (const canonical::activity_t&, USER_TYPE& ),
                   canonical::activity_t (*u2k)( const USER_TYPE&) )
    : kpf_io_adapter<USER_TYPE, canonical::activity_t>( k2u, u2k )
  {
    this->packet_bounce.init( packet_header_t( packet_style::ACT ));
  }

  USER_TYPE get()
  {
    auto probe = this->packet_bounce.get_packet();
    // throw if ! probe->first
    // also throw if kpf2user is null, or else use a temporary?
    return (this->kpf2user_function)( probe.second.activity );
  }
  void get( USER_TYPE& u )
  {
    auto probe = this->packet_bounce.get_packet();
    // see above
    (this->kpf2user_inplace)( probe.second.activity, u );
  }

  void get( kpf_reader_t& parser, USER_TYPE& u )
  {
    // same throwing issues
    parser.process( *this );
    this->get( u );
  }

  canonical::activity_t operator()( const USER_TYPE& u )
  {
    return this->user2kpf_function( u );
  }

};

} // ...kpf
} // ...vital
} // ...kwiver

#endif
