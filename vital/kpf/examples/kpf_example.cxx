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

//
// This is a notional data structure that holds object detections.
// The object detections have an ID, a frame number, a box, a label,
// and a confidence value.
//
// The box is stored as a corner point and a width and height.
//

struct whizbang_detection_t
{
  int detection_id;
  unsigned frame_number;
  pair< double, double > box_corner_pt;
  double box_width;
  double box_height;
  string label;
  double confidence;
  whizbang_detection_t()
    : detection_id(0), frame_number(0), box_corner_pt( {0,0}),
      box_width(0), box_height(0), label("invalid"), confidence(0)
  {}
  whizbang_detection_t( int d, unsigned f, const pair<double, double>& c,
                        double w, double h, const string& s, double conf )
    : detection_id(d), frame_number(f), box_corner_pt(c),
      box_width(w), box_height(h), label(s), confidence(conf)
  {}
};

ostream& operator<<( ostream& os, const whizbang_detection_t& d )
{
  os << "detection " << d.detection_id << " @ frame " << d.frame_number << ": "
     << d.box_width << "x" << d.box_height << "+" << d.box_corner_pt.first
     << "+" << d.box_corner_pt.second << "; label '" << d.label << "' conf "
     << d.confidence;
  return os;
}


//
// Generate some sample detections.
//

vector< whizbang_detection_t >
make_sample_detections()
{
  vector< whizbang_detection_t > detections {
    { 100, 4, { 33.3, 33.3 }, 10, 20, "vehicle", 0.3 },
    { 101, 4, { 44.4, 44.4 }, 4, 9,   "person",  0.8 },
    { 102, 5, { 55.5, 55.5 }, 11, 7,  "vehicle", 0.5 }
  };

  return detections;
}


//
// For non-scalar data which is represented by a non-scalar KPF structure
// (e.g. a bounding box), you need to define two routines: one converts
// your structure into KPF, the other converts KPF into your structure.
//

//
// The KPF 'canonical' bounding box is (x1,y1)-(x2,y2).
//

struct whizbang_box_adapter_t: public KPF::kpf_box_adapter< whizbang_detection_t >
{
  whizbang_box_adapter_t( int domain ):
    kpf_box_adapter< whizbang_detection_t >(
      // this lambda reads the canonical box into the existing whizbang_detection
      []( const KPF::canonical::bbox_t& b, whizbang_detection_t& d ) {
        d.box_corner_pt = make_pair( b.x1, b.y1 );
        d.box_width = (b.x2-b.x1);
        d.box_height = (b.y2-b.y1); },

      // this lambda converts a whizbang_detection into a canonical box
      []( const whizbang_detection_t& d ) {
        return KPF::canonical::bbox_t(
          d.box_corner_pt.first,
          d.box_corner_pt.second,
          d.box_corner_pt.first + d.box_width,
          d.box_corner_pt.second + d.box_height );},

      domain)
  {}
};


//
// Read a set of detections from a stream.
//
// Note that we're implicitly expecting to find one per line.
//

vector< whizbang_detection_t >
read_detections_from_stream( std::istream& is )
{
  vector< whizbang_detection_t > dets;
  whizbang_box_adapter_t box(0); // from domain 0;
  KPF::text_parser_t parser( is );

  while (parser >> box)
  {
    whizbang_detection_t det;
    // have to use this form since it matches the lambda
    box.get( det );
    dets.push_back( det );
  }
  return dets;
}

//
// Write a set of detections to a stream as KPF.
//

void
write_detections_to_stream( ostream& os,
                            const vector< whizbang_detection_t >& dets )
{
  whizbang_box_adapter_t box(0); // to domain 0;
  for (size_t i=0; i<dets.size(); ++i)
  {
    os
      << box.to_str( dets[i] )
      << "\n";
  }
}

int main()
{

  vector< whizbang_detection_t > src_dets = make_sample_detections();
  for (auto i=0; i<src_dets.size(); ++i)
  {
    std::cout << "Source det " << i << ": " << src_dets[i] << "\n";
  }

  stringstream ss;
  std::cout << "\nAbout to write detections:\n";
  write_detections_to_stream( ss, src_dets );
  std::cout << "KPF representation:\n" << ss.str();
  std::cout << "Done\n";

  std::cout << "\nAbout to read KPF:\n";
  vector< whizbang_detection_t> new_dets = read_detections_from_stream( ss );
  for (auto i=0; i<src_dets.size(); ++i)
  {
    std::cout << "Converted det " << i << ": " << new_dets[i] << "\n";
  }


}
