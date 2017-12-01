/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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

