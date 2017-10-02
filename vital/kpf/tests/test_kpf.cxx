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
    KPF::text_record_reader_t r( std::cin );
    TEST_EQUAL( "Empty reader is empty", r.get_packet_buffer().empty(), true );
  }

  /*
  {
    string simple_box_str("G0: 10 10 20 20");
    istringstream iss( simple_box_str );
    KPF::text_record_reader_t r( iss );
    KPF::null_reader_t null_reader;

    //
    // Note that 'bool okay = (r >> null_reader)' doesn't work, since
    // explicit conversion doesn't work with copy initialization
    //
    bool okay(r >> null_reader);

    TEST_EQUAL( "Reading one box: null reader succeeds", okay, true );
    TEST_EQUAL( "Reading one box: one packet in the buffer", r.get_packet_buffer().size(), 1 );
    KPF::packet_header_t g0( KPF::packet_style::GEOM, 0 );
    auto g0_probe = r.get_packet_buffer().find( g0 );
    TEST_EQUAL( "Reading one box: packet is g0", g0_probe != r.get_packet_buffer().end(), true );
  }
  */
}

IMPLEMENT_TEST( basic_kpf_text_parsing )
{
  map< string, KPF::header_parse_t > tests;

  tests.insert( make_pair( "", make_tuple( false, string(), KPF::packet_header_t::NO_DOMAIN )));
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
    KPF::header_parse_t result = KPF::parse_header( i->first );

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
