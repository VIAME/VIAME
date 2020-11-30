// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief KPF YAML read/write test.
 *
 * Adapted from examples/cpp/kpf/kpf_example_complex.cxx
 *
 */

#include <test_common.h>

#include <arrows/kpf/yaml/kpf_reader.h>
#include <arrows/kpf/yaml/kpf_yaml_parser.h>
#include <arrows/kpf/yaml/kpf_yaml_writer.h>
#include <arrows/kpf/yaml/kpf_canonical_io_adapter.h>

#include <string>
#include <sstream>
#include <map>
#include <utility>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

using std::string;
using std::stringstream;
using std::istringstream;
using std::ostringstream;
using std::ostream;
using std::vector;
using std::map;
using std::pair;
using std::make_pair;
using std::make_tuple;

namespace KPF=kwiver::vital::kpf;
namespace KPFC=kwiver::vital::kpf::canonical;

namespace { // anon

const int DETECTOR_DOMAIN = 17;

struct user_complex_detection_t
{
  size_t detection_id;
  unsigned frame_number;
  pair< double, double > box_corner_pt;
  double box_width;
  double box_height;
  string label;
  double confidence;
  vector< double > poly_x, poly_y;
  user_complex_detection_t()
    : detection_id(0), frame_number(0), box_corner_pt( {0,0}),
      box_width(0), box_height(0), label("invalid"), confidence(0)
  {}
  user_complex_detection_t( int d, unsigned f, const pair<double, double>& c,
                    double w, double h, const string& s, double conf,
                    const vector<double>& x, const vector<double>& y)
    : detection_id(d), frame_number(f), box_corner_pt(c),
      box_width(w), box_height(h), label(s), confidence(conf), poly_x(x), poly_y(y)
  {}
};

bool operator==( const user_complex_detection_t& lhs,
                 const user_complex_detection_t& rhs )
{
  return
    (lhs.detection_id == rhs.detection_id) &&
    (lhs.frame_number == rhs.frame_number) &&
    (lhs.box_corner_pt.first == rhs.box_corner_pt.first) &&
    (lhs.box_corner_pt.second == rhs.box_corner_pt.second) &&
    (lhs.box_width == rhs.box_width) &&
    (lhs.box_height == rhs.box_height) &&
    (lhs.label == rhs.label) &&
    (lhs.confidence == rhs.confidence) &&
    (lhs.poly_x == rhs.poly_x) &&
    (lhs.poly_y == rhs.poly_y);
}

struct user_box_adapter_t: public KPF::kpf_box_adapter< user_complex_detection_t >
{
  user_box_adapter_t():
    kpf_box_adapter< user_complex_detection_t >(
      // reads the canonical box "b" into the user_detection "d"
      []( const KPF::canonical::bbox_t& b, user_complex_detection_t& d ) {
        d.box_corner_pt = make_pair( b.x1, b.y1 );
        d.box_width = (b.x2-b.x1);
        d.box_height = (b.y2-b.y1); },

      // converts a user_detection "d" into a canonical box and returns it
      []( const user_complex_detection_t& d ) {
        return KPF::canonical::bbox_t(
          d.box_corner_pt.first,
          d.box_corner_pt.second,
          d.box_corner_pt.first + d.box_width,
          d.box_corner_pt.second + d.box_height );} )
  {}
};

//
// This adapter converts the KPF polygon structure into the user's
// polygon structure.
//

struct user_poly_adapter_t: public KPF::kpf_poly_adapter< user_complex_detection_t >
{
  user_poly_adapter_t():
    kpf_poly_adapter< user_complex_detection_t >(
      // reads the canonical box "b" into the user_detection "d"
      []( const KPF::canonical::poly_t& b, user_complex_detection_t& d ) {
        d.poly_x.clear(); d.poly_y.clear();
        for (auto p: b.xy) {
          d.poly_x.push_back(p.first);
          d.poly_y.push_back(p.second);
        }},
      // converts a user_detection "d" into a canonical box and returns it
      []( const user_complex_detection_t& d ) {
        KPF::canonical::poly_t p;
        // should check that d's vectors are the same length
        for (size_t i=0; i<d.poly_x.size(); ++i) {
          p.xy.push_back( make_pair( d.poly_x[i], d.poly_y[i] ));
        }
        return p; })
  {}
};

//
// Generate some sample detections.
//

vector< user_complex_detection_t >
make_sample_detections()
{
  return {
    { 100, 4, { 33.3, 33.3 }, 10, 20, "vehicle", 0.3, {10,20,10}, {10,20,30}},
    { 101, 4, { 44.4, 44.4 }, 4, 9,   "person",  0.8, {10,20,20,10},{10,10,20,20}},
    { 102, 5, { 55.5, 55.5 }, 11, 7,  "vehicle", 0.5, {1,2,1},{1,2,3}}
  };
}

vector< user_complex_detection_t >
read_detections_from_stream( std::istream& is )
{
  namespace KPFC = KPF::canonical;
  vector< user_complex_detection_t > dets;
  user_box_adapter_t box;
  user_poly_adapter_t poly;

  KPF::kpf_yaml_parser_t parser( is );
  KPF::kpf_reader_t reader( parser );

  // each record will be read into a buffer object
  user_complex_detection_t buffer;

  //
  // Here the reader object populates the adapters with their respective
  // structures, but the user must explicitly call get() on the adapter
  // to copy it into the buffer before copying the buffer into the vector.
  //

  while (reader
         >> KPF::reader< KPFC::bbox_t >( box, KPFC::bbox_t::IMAGE_COORDS )
         >> KPF::reader< KPFC::id_t >( buffer.detection_id, KPFC::id_t::DETECTION_ID )
         >> KPF::reader< KPFC::timestamp_t>( buffer.frame_number, KPFC::timestamp_t::FRAME_NUMBER )
         >> KPF::reader< KPFC::kv_t>( "label", buffer.label )
         >> KPF::reader< KPFC::conf_t>( buffer.confidence, DETECTOR_DOMAIN )
         >> KPF::reader< KPFC::poly_t >( poly, KPFC::poly_t::IMAGE_COORDS )
    )
  {
    box.get( buffer );
    poly.get( buffer );
    dets.push_back( buffer );

    //
    // Metadata packets can appear anywhere in the stream. The reader object
    // buffers them up until it sees the next non-metadata record (or end-of-file.)
    //

    reader.flush();
  }

  return dets;
}

void
write_detections_to_stream( ostream& os,
                            const vector< user_complex_detection_t >& dets )
{
  namespace KPFC = KPF::canonical;
  user_box_adapter_t box_adapter;
  user_poly_adapter_t poly_adapter;
  KPF::record_yaml_writer w( os );
  size_t line_count = 0;
  for (const auto& det: dets )
  {
    //
    // Generate some gratuitous metadata; write it out as its own record.
    //

    ostringstream oss;
    oss << "Record " << line_count++;
    w
      << KPF::writer< KPFC::meta_t >( oss.str() )
      << KPF::record_yaml_writer::endl;

    //
    // Write out the actual detection.
    //

    w
      << KPF::writer< KPFC::bbox_t >( box_adapter( det ), KPFC::bbox_t::IMAGE_COORDS )
      << KPF::writer< KPFC::id_t >( det.detection_id, KPFC::id_t::DETECTION_ID )
      << KPF::writer< KPFC::timestamp_t >( det.frame_number, KPFC::timestamp_t::FRAME_NUMBER )
      << KPF::writer< KPFC::poly_t>( poly_adapter( det ), KPFC::poly_t::IMAGE_COORDS )
      << KPF::writer< KPFC::kv_t >( "label", det.label )
      << KPF::writer< KPFC::conf_t>( det.confidence, DETECTOR_DOMAIN )
      << KPF::record_yaml_writer::endl;
  }
}

static const string improper_kpf =
         "- { geom: { src: ground-truth, g0: 1751 223 1892 652, id0: 1, id1: 1, ts0: 2406, ts1: 0 } }\n"
         "- { geom: { src: ground-truth, g0: 1349 299 1563 929, id0: 2, id1: 1, ts0: 6873, ts1: 0 } }\n"
         "- { geom: { src: truth, g0: 1733 212 1888 664, occlusion: , id0: 3, id1: 1, ts0: 2407, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1728 213 1884 667, occlusion: , id0: 4, id1: 1, ts0: 2408, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1723 214 1881 670, occlusion: , id0: 5, id1: 1, ts0: 2409, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1718 215 1877 673, occlusion: , id0: 6, id1: 1, ts0: 2410, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1713 216 1873 676, occlusion: , id0: 7, id1: 1, ts0: 2411, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1709 216 1870 679, occlusion: , id0: 8, id1: 1, ts0: 2412, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1704 217 1866 682, occlusion: , id0: 9, id1: 1, ts0: 2413, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1699 218 1862 685, occlusion: , id0: 10, id1: 1, ts0: 2414, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1694 219 1859 688, occlusion: , id0: 11, id1: 1, ts0: 2415, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1689 220 1855 691, occlusion: , id0: 12, id1: 1, ts0: 2416, ts1: 0 } }\n"
         "- { geom: { src: truth, g0: 1689 220 1855 691, occlusion: , id0: 13, id1: 1, ts0: 2417, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1681 221 1851 692, occlusion: , id0: 14, id1: 1, ts0: 2418, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1673 222 1847 693, occlusion: , id0: 15, id1: 1, ts0: 2419, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1665 222 1844 694, occlusion: , id0: 16, id1: 1, ts0: 2420, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1657 223 1840 695, occlusion: , id0: 17, id1: 1, ts0: 2421, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1649 224 1836 696, occlusion: , id0: 18, id1: 1, ts0: 2422, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1641 225 1832 697, occlusion: , id0: 19, id1: 1, ts0: 2423, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1633 225 1829 698, occlusion: , id0: 20, id1: 1, ts0: 2424, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1625 226 1825 699, occlusion: , id0: 21, id1: 1, ts0: 2425, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1617 227 1821 700, occlusion: , id0: 22, id1: 1, ts0: 2426, ts1: 0 } }\n"
         "- { geom: { src: truth, g0: 1617 227 1821 700, occlusion: , id0: 23, id1: 1, ts0: 2427, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1615 227 1821 706, occlusion: , id0: 24, id1: 1, ts0: 2428, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1613 227 1821 711, occlusion: , id0: 25, id1: 1, ts0: 2429, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1611 227 1820 717, occlusion: , id0: 26, id1: 1, ts0: 2430, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1609 227 1820 722, occlusion: , id0: 27, id1: 1, ts0: 2431, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1606 228 1820 728, occlusion: , id0: 28, id1: 1, ts0: 2432, ts1: 0 } }\n"
         "- { geom: { src: linear-interpolation, g0: 1604 228 1820 733, occlusion: , id0: 29, id1: 1, ts0: 2433, ts1: 0 } }\n";

} // ... anon

int
main( int argc, char* argv[] )
{
  CHECK_ARGS(1);

  const testname_t testname = argv[1];

  RUN_TEST( testname );
}

IMPLEMENT_TEST( kpf_yaml_complex_io )
{
  vector< user_complex_detection_t > src_dets = make_sample_detections();
  stringstream ss;
  write_detections_to_stream( ss, src_dets );
  vector< user_complex_detection_t> new_dets = read_detections_from_stream( ss );
  {
    ostringstream oss;
    oss << "Wrote " << src_dets.size() << " detections; read " << new_dets.size() << " back";
    TEST_EQUAL( oss.str(), src_dets.size() == new_dets.size(), true );
  }

  for (size_t i=0; i<src_dets.size(); ++i)
  {
    ostringstream oss;
    oss << "Detection " << i << ": source eq output";
    TEST_EQUAL( oss.str(), src_dets[i] == new_dets[i], true );
  }
}

IMPLEMENT_TEST( improper_kpf_parse )
{
  istringstream iss( improper_kpf );

  KPF::kpf_yaml_parser_t parser( iss );
  KPF::kpf_reader_t reader( parser );

  while (reader)
  {
    reader.next();
    reader.flush();
  }
  TEST_EQUAL( "Got to end of improper kpf_parse", true, true );
}

IMPLEMENT_TEST( kpf_packet_extraction )
{
  vector< user_complex_detection_t > src_dets = make_sample_detections();
  stringstream ss;
  write_detections_to_stream( ss, src_dets );

  KPF::kpf_yaml_parser_t parser( ss );
  KPF::kpf_reader_t reader( parser );

  TEST_EQUAL( "Start of packet extraction: reader is good", static_cast<bool>(reader), true );

  // read a line of packets
  reader.next();
  TEST_EQUAL( "Read one line; reader is good", static_cast<bool>(reader), true );

  auto packet_buffer = reader.get_packet_buffer();
  TEST_EQUAL( "After one line: non-zero number of packets", packet_buffer.empty(), false );

  const auto id_header = KPF::packet_header_t( KPF::packet_style::ID, KPFC::id_t::DETECTION_ID );

  // get the detection ID
  auto id_one = reader.transfer_packet_from_buffer( id_header );
  TEST_EQUAL( "After one line: first transfer of ID succeeded", id_one.first, true );
  TEST_EQUAL( "After one line: reader still good", static_cast<bool>(reader), true );

  // get the detection ID again (should fail but leave reader good)
  auto id_two = reader.transfer_packet_from_buffer( id_header );
  TEST_EQUAL( "After one line: second transfer of ID failed", id_two.first, false );
  TEST_EQUAL( "After one line: reader still good", static_cast<bool>(reader), true );

  // we should be able to read non-zero more lines
  size_t c = 0;
  while (reader.next())
  {
    ++c;
    reader.flush();
  }
  TEST_EQUAL( "Read more lines after first line", c > 0, true );

}
