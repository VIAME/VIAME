//
// An example of how one uses KPF to read and write custom data structures.
//


#include <vital/kpf/kpf_parse.h>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using std::string;
using std::vector;
using std::pair;
using std::make_pair;
using std::istringstream;
using std::stringstream;
using std::ostream;

namespace KPF=kwiver::vital::kpf;


struct user_detection_t
{
  int detection_id;
  unsigned frame_number;
  pair< double, double > box_corner_pt;
  double box_width;
  double box_height;
  string label;
  double confidence;
  vector< double > poly_x, poly_y;
  user_detection_t()
    : detection_id(0), frame_number(0), box_corner_pt( {0,0}),
      box_width(0), box_height(0), label("invalid"), confidence(0)
  {}
  user_detection_t( int d, unsigned f, const pair<double, double>& c,
                    double w, double h, const string& s, double conf,
                    const vector<double>& x, const vector<double>& y)
    : detection_id(d), frame_number(f), box_corner_pt(c),
      box_width(w), box_height(h), label(s), confidence(conf), poly_x(x), poly_y(y)
  {}
};

//
// pretty-print the user detections
//

ostream& operator<<( ostream& os, const user_detection_t& d )
{
  os << "detection " << d.detection_id << " @ frame " << d.frame_number << ": "
     << d.box_width << "x" << d.box_height << "+" << d.box_corner_pt.first
     << "+" << d.box_corner_pt.second << "; label '" << d.label << "' conf "
     << d.confidence << " polygon w/ " << d.poly_x.size() << " points";
  return os;
}

//
// Generate some sample detections.
//

vector< user_detection_t >
make_sample_detections()
{
  return {
    { 100, 4, { 33.3, 33.3 }, 10, 20, "vehicle", 0.3, {10,20,10}, {10,20,30}},
    { 101, 4, { 44.4, 44.4 }, 4, 9,   "person",  0.8, {10,20,20,10},{10,10,20,20}},
    { 102, 5, { 55.5, 55.5 }, 11, 7,  "vehicle", 0.5, {1,2,1},{1,2,3}}
  };
}


//
// This is the socially-agreed domain for the detector; an arbitrary number
// greater than 9 which disambiguates this detector from others we may
// have on the project.
//

const int DETECTOR_DOMAIN=17;

//
// For non-scalar data which is represented by a non-scalar KPF structure
// (e.g. a bounding box), you need to define two routines: one converts
// your structure into KPF, the other converts KPF into your structure.
//
// Note that since the box is "distributed" across several fields of
// user_detection_t, the adapter needs to take an instance of the
// complete user type, even though it's only setting a few fields.

//
// The KPF 'canonical' bounding box is (x1,y1)-(x2,y2).
//

struct user_box_adapter_t: public KPF::kpf_box_adapter< user_detection_t >
{
  user_box_adapter_t():
    kpf_box_adapter< user_detection_t >(
      // reads the canonical box "b" into the user_detection "d"
      []( const KPF::canonical::bbox_t& b, user_detection_t& d ) {
        d.box_corner_pt = make_pair( b.x1, b.y1 );
        d.box_width = (b.x2-b.x1);
        d.box_height = (b.y2-b.y1); },

      // converts a user_detection "d" into a canonical box and returns it
      []( const user_detection_t& d ) {
        return KPF::canonical::bbox_t(
          d.box_corner_pt.first,
          d.box_corner_pt.second,
          d.box_corner_pt.first + d.box_width,
          d.box_corner_pt.second + d.box_height );} )
  {}
};

struct user_poly_adapter_t: public KPF::kpf_poly_adapter< user_detection_t >
{
  user_poly_adapter_t():
    kpf_poly_adapter< user_detection_t >(
      // reads the canonical box "b" into the user_detection "d"
      []( const KPF::canonical::poly_t& b, user_detection_t& d ) {
        d.poly_x.clear(); d.poly_y.clear();
        for (auto p: b.xy) {
          d.poly_x.push_back(p.first);
          d.poly_y.push_back(p.second);
        }},
      // converts a user_detection "d" into a canonical box and returns it
      []( const user_detection_t& d ) {
        KPF::canonical::poly_t p;
        // should check that d's vectors are the same length
        for (size_t i=0; i<d.poly_x.size(); ++i) {
          p.xy.push_back( make_pair( d.poly_x[i], d.poly_y[i] ));
        }
        return p; })
  {}
};

//
// Read a set of detections from a stream.
//
// Note that we're implicitly expecting to find record one per line.
//

vector< user_detection_t >
read_detections_from_stream( std::istream& is )
{
  namespace KPFC = KPF::canonical;
  vector< user_detection_t > dets;
  user_box_adapter_t box;
  user_poly_adapter_t poly;
  KPF::text_parser_t parser( is );
  user_detection_t buffer;

  while (parser
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
    parser.flush();
  }


  return dets;
}

void
write_detections_to_stream( ostream& os,
                            const vector< user_detection_t >& dets )
{
  namespace KPFC = KPF::canonical;
  user_box_adapter_t box_adapter;
  user_poly_adapter_t poly_adapter;
  KPF::record_text_writer w( os );
  for (const auto& det: dets )
  {
    w
      << KPF::writer< KPFC::bbox_t >( box_adapter( det ), KPFC::bbox_t::IMAGE_COORDS )
      << KPF::writer< KPFC::id_t >( det.detection_id, KPFC::id_t::DETECTION_ID )
      << KPF::writer< KPFC::timestamp_t >( det.frame_number, KPFC::timestamp_t::FRAME_NUMBER )
      << KPF::writer< KPFC::kv_t >( "label", det.label )
      << KPF::writer< KPFC::conf_t>( det.confidence, DETECTOR_DOMAIN )
      << KPF::writer< KPFC::poly_t>( poly_adapter( det ), KPFC::poly_t::IMAGE_COORDS )
      << KPF::record_text_writer::endl;
  }
}

int main()
{

  vector< user_detection_t > src_dets = make_sample_detections();
  for (auto i=0; i<src_dets.size(); ++i)
  {
    std::cout << "Source det " << i << ": " << src_dets[i] << "\n";
  }

  stringstream ss;
  std::cout << "\nAbout to write detections (with metadata):\n";
  write_detections_to_stream( ss, src_dets );
  ss << "meta: generated from " << __FILE__ << "\n";
  std::cout << "KPF representation:\n" << ss.str();
  std::cout << "Done\n";

  std::cout << "\nAbout to read KPF:\n";
  vector< user_detection_t> new_dets = read_detections_from_stream( ss );
  for (auto i=0; i<new_dets.size(); ++i)
  {
    std::cout << "Converted det " << i << ": " << new_dets[i] << "\n";
  }


}
