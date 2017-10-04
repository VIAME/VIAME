#include <test_common.h>

#include <vital/kpf/kpf_parse.h>
#include <vital/kpf/kpf_parse_utils.h>
#include <vital/logger/logger.h>

#include <string>
#include <sstream>
#include <map>
#include <utility>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

using std::string;
using std::istringstream;
using std::ostringstream;
using std::map;
using std::make_pair;
using std::make_tuple;

static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

namespace KPF=kwiver::vital::kpf;

namespace { // anon

bool operator==( const KPF::canonical::bbox_t& lhs,
                 const KPF::canonical::bbox_t& rhs )
{
  return (lhs.x1 == rhs.x1) && (lhs.y1 == rhs.y1) && (lhs.x2 == rhs.x2) && (lhs.y2 == rhs.y2);
}

} // ... anon


int
main( int argc, char* argv[] )
{
  CHECK_ARGS(1);

  const testname_t testname = argv[1];

  RUN_TEST( testname );
}

IMPLEMENT_TEST( basic_kpf_from_string )
{
  {
    // sanity check
    KPF::text_parser_t r( std::cin );
    TEST_EQUAL( "Empty reader is empty", r.get_packet_buffer().empty(), true );
  }

  {
    map< string, bool > tests;
    tests.insert( make_pair("g0:", false ));
    tests.insert( make_pair("g0: 10 ", false ));
    tests.insert( make_pair("g0: 10 20", false ));
    tests.insert( make_pair("g0: 10 20 30", false ));
    tests.insert( make_pair("g0: 10 20 30 40", true ));
    tests.insert( make_pair("g0: 10 x 30 40", false ));
    tests.insert( make_pair("g0: 0.1 -1.0e-06 3.14159 10e9", true ));

    for (auto t = tests.begin(); t != tests.end(); ++t)
    {
      istringstream iss( t->first );
      KPF::text_parser_t r( iss );
      KPF::text_reader_t reader( "g0" );

      ostringstream oss;
      if (! t->second )
      {
        LOG_ERROR( main_logger, "The following test should generate an error" );
        oss << "Reading one box '" << t->first << "': reader should fail";
      }
      else
      {
        oss << "Reading one box '" << t->first << "': reader should succeed";
      }

      //
      // Note that 'bool okay = (r >> reader)' doesn't work, since
      // explicit conversion doesn't work with copy initialization
      //
      bool okay(r >> reader);
      TEST_EQUAL( "Reading one box: reader succeeds", okay, t->second );
      if (t->second)
      {
        auto g0_probe = reader.get_packet();
        TEST_EQUAL( "Reading one box: reader has packet", g0_probe.first, true );
        KPF::packet_t packet = g0_probe.second;
        TEST_EQUAL( "Reading one box: packet is geom", packet.header.style == KPF::packet_style::GEOM, true );
        TEST_EQUAL( "Reading one box: packet domain is 0", packet.header.domain, 0 );
      }
    }
  }
}

IMPLEMENT_TEST( kpf_box_reader )
{
  {
    KPF::canonical::bbox_t ref( 10, 20, 30, 40 );
    istringstream iss( "g0: 10 20 30 40" );
    KPF::text_reader_t reader( "g0" );
    KPF::text_parser_t r( iss );

    {
    auto probe = reader.get_packet();
      TEST_EQUAL( "Reader is invalid before read", probe.first, false );
    }

    bool okay(r >> reader);
    TEST_EQUAL( "Reading a box succeeds", okay, true );

    {
      auto probe = reader.get_packet();
      TEST_EQUAL( "Reader is valid after read", probe.first, true );
      TEST_EQUAL( "Packet's box equals input", probe.second.payload.bbox == ref, true );
    }
  }

  {
    istringstream iss( "  g0: 1 1 1 1 g0: 2 2 2 2 g0: 3 3 3 3 g0: 4 4 4 4   " );
    KPF::text_reader_t reader( "g0" );
    KPF::text_parser_t r( iss );
    const size_t N_boxes = 4;
    for (size_t i=0; i<N_boxes+1; ++i)
    {
      bool okay( r >> reader );
      bool expected = (i < N_boxes);
      ostringstream oss;
      oss << "Group box read " << i+1 << " of " << N_boxes+1+1;
      TEST_EQUAL( oss.str(), okay, expected );
    }
  }
}

IMPLEMENT_TEST( basic_kpf_text_parsing )
{
  map< string, KPF::header_parse_t > tests;

  tests.insert( make_pair( "", make_tuple( false, string(), KPF::packet_header_t::NO_DOMAIN )));
  tests.insert( make_pair( ":", make_tuple( false, string(), KPF::packet_header_t::NO_DOMAIN )));
  tests.insert( make_pair( "1:", make_tuple( false, string(), KPF::packet_header_t::NO_DOMAIN )));
  tests.insert( make_pair( "100:", make_tuple( false, string(), KPF::packet_header_t::NO_DOMAIN )));
  tests.insert( make_pair( "G1", make_tuple( false, string(), KPF::packet_header_t::NO_DOMAIN )));
  tests.insert( make_pair( "G1:", make_tuple( true, "G", 1 )));
  tests.insert( make_pair( "meta:", make_tuple( true, "meta", KPF::packet_header_t::NO_DOMAIN )));
  tests.insert( make_pair( "m:", make_tuple( true, "m", KPF::packet_header_t::NO_DOMAIN )));
  tests.insert( make_pair( "long_tag9:", make_tuple( true, "long_tag", 9)));
  tests.insert( make_pair( "s12345:", make_tuple( true, "s", 12345)));
  tests.insert( make_pair( "eval15:", make_tuple( true, "eval", 15)));

  for (auto i=tests.begin(); i != tests.end(); ++i)
  {
    bool okay_expected, okay_obtained;
    string tag_expected, tag_obtained;
    int domain_expected, domain_obtained;
    std::tie( okay_expected, tag_expected, domain_expected) = i->second;

    if (! okay_expected)
    {
      LOG_ERROR( main_logger, "Following call to parse_header is expected to induce an error" );
    }
    KPF::header_parse_t result = KPF::parse_header( i->first, true );

    std::tie( okay_obtained, tag_obtained, domain_obtained) = result;
    ostringstream oss;
    oss << "Parsing tag '" << i->first << "': expecting "
        << okay_expected << " '" << tag_expected << "' @ " << domain_expected
        << "; obtained "
        << okay_obtained << " '" << tag_obtained << "' @ " << domain_obtained;
    TEST_EQUAL( oss.str()+": okay ", okay_expected, okay_obtained );
    TEST_EQUAL( oss.str()+": tag ", tag_expected, tag_obtained );
    TEST_EQUAL( oss.str()+": domain ", domain_expected, domain_obtained );
  }
}
