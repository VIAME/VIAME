// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Simple example of reading / writing KPF with object classifications.
 *
 */

#include <arrows/kpf/yaml/kpf_reader.h>
#include <arrows/kpf/yaml/kpf_yaml_parser.h>
#include <arrows/kpf/yaml/kpf_yaml_writer.h>

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
// This is our simple detection object; our classifier knows about 'person',
// 'vehicle', and 'other' categories.
//

struct user_simple_detection_t
{
  size_t detection_id;
  unsigned frame_number;
  vector<double> confidence; // 0 = person, 1 = vehicle, 2 = other
  user_simple_detection_t()
    : detection_id(0), frame_number(0), confidence( {0, 0, 0 })
  {}
  user_simple_detection_t( int d, unsigned f, const vector<double>& conf )
    : detection_id(d), frame_number(f), confidence(conf)
  {}
  static vector<string> conf_tags() { return {"person", "vehicle", "other"}; }
};

//
// pretty-print the user detections
//

ostream& operator<<( ostream& os, const user_simple_detection_t& d )
{
  os << "detection " << d.detection_id << " @ frame " << d.frame_number
     << ": p-conf " << d.confidence[0] << "; v-conf " << d.confidence[1]
     << "; o-conf " << d.confidence[2];
  return os;
}

//
// Generate some sample detections.
//

vector< user_simple_detection_t >
make_sample_detections()
{
  return {
    { 100, 4, {0.3, 0.6, 0.1} },
    { 101, 4, {0.8, 0.15, 0.05} },
    { 102, 5, {0.5, 0.2, 0.3} }
  };
}

//
// This is the socially-agreed domain for the detector; an arbitrary number
// greater than 9 which disambiguates this detector from others we may
// have on the project.
//

const int DETECTOR_DOMAIN=17;

//
// Read a set of detections from a stream.
//
// Note that we're implicitly expecting to find record one per line.
//

vector< user_simple_detection_t >
read_detections_from_stream( std::istream& is )
{
  namespace KPFC = KPF::canonical;

  // read them into this vector
  vector< user_simple_detection_t > dets;

  // associate the input stream with the YAML parser
  KPF::kpf_yaml_parser_t parser( is );

  // associate a KPF reader with the parser
  KPF::kpf_reader_t reader( parser );

  // allocate an instance of the user type as a read buffer
  user_simple_detection_t buffer;

  //
  // The reader collects all the packets for each record; the KPF::reader< t > calls
  // select the packets based on type and domain and copy them into the particular
  // members of the buffer object.
  //
  // Note that the *order* in which the packets appear in the KPF record doesn't
  // matter. (The order in which the *records* (lines) appear DOES matter.)
  //
  // Here we'll just use a buffer of KPF's canonical classifier type, a map
  // of string->double, in order to read our classifier values.
  //

  KPFC::cset_t conf_map;

  const auto& conf_tags = user_simple_detection_t::conf_tags();
  while (reader
         >> KPF::reader< KPFC::id_t >( buffer.detection_id, KPFC::id_t::DETECTION_ID )
         >> KPF::reader< KPFC::timestamp_t>( buffer.frame_number, KPFC::timestamp_t::FRAME_NUMBER )
         >> KPF::reader< KPFC::cset_t>( conf_map, DETECTOR_DOMAIN )
    )
  {
    // move the confidence values in
    for ( size_t i=0; i<conf_tags.size(); ++i)
    {
      const auto& tag = conf_tags[i];
      auto probe = conf_map.d.find( tag );
      double v = 0.0;
      if (probe == conf_map.d.end())
      {
        std::cout << "No confidence value found for '" << tag << "; assuming zero\n";
      }
      else
      {
        v = probe->second;
      }
      buffer.confidence[ i ] = v;
    }

    dets.push_back( buffer );

    // the flush call throws away any optional packets we didn't copy into our buffer
    reader.flush();
  }
  return dets;
}

//
// Write a set of detections to a stream as KPF.
//

void
write_detections_to_stream( ostream& os,
                            const vector< user_simple_detection_t >& dets )
{
  namespace KPFC = KPF::canonical;

  // associate a yaml writer with the output stream
  KPF::record_yaml_writer w( os );

  //
  // each detection becomes a KPF record; its contents become packets.
  //

  // Here, we explicitly leave the schema unspecified.
  const auto& conf_tags = user_simple_detection_t::conf_tags();

  for (const auto& det: dets )
  {
    //
    // set up a KPF confidence set to transfer the confidence values.
    //

    KPFC::cset_t conf_map;
    for (size_t i=0; i<conf_tags.size(); ++i)
    {
      conf_map.d.insert( make_pair( conf_tags[i], det.confidence[i] ));
    }

    w.set_schema( KPF::schema_style::UNSPECIFIED )
      << KPF::writer< KPFC::id_t >( det.detection_id, KPFC::id_t::DETECTION_ID )
      << KPF::writer< KPFC::timestamp_t >( det.frame_number, KPFC::timestamp_t::FRAME_NUMBER )
      << KPF::writer< KPFC::cset_t>( conf_map, DETECTOR_DOMAIN )
      << KPF::record_yaml_writer::endl;
  }
}

int main()
{

  vector< user_simple_detection_t > src_dets = make_sample_detections();
  std::cout << "\n";
  for (size_t i=0; i<src_dets.size(); ++i)
  {
    std::cout << "Source det " << i << ": " << src_dets[i] << "\n";
  }

  stringstream ss;
  std::cout << "\nAbout to write detections:\n";
  write_detections_to_stream( ss, src_dets );
  std::cout << "KPF representation:\n" << ss.str();
  std::cout << "Done\n";

  std::cout << "\nAbout to read KPF:\n";
  vector< user_simple_detection_t> new_dets = read_detections_from_stream( ss );
  for (size_t i=0; i<new_dets.size(); ++i)
  {
    std::cout << "Converted det " << i << ": " << new_dets[i] << "\n";
  }
}
