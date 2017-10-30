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
 * \brief Simple example of reading / writing KPF for simple packets.
 *
 */


#include <vital/kpf/kpf_reader.h>
#include <vital/kpf/kpf_text_parser.h>
#include <vital/kpf/kpf_text_writer.h>

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
// -----------------------------
// -----------------------------
//
// PART ONE: The user type we'll be playing with
//
// -----------------------------
// -----------------------------
//


struct user_detection_t
{
  int detection_id;
  unsigned frame_number;
  double confidence;
  user_detection_t()
    : detection_id(0), frame_number(0), confidence(0)
  {}
  user_detection_t( int d, unsigned f, double conf )
    : detection_id(d), frame_number(f), confidence(conf)
  {}
};

//
// pretty-print the user detections
//

ostream& operator<<( ostream& os, const user_detection_t& d )
{
  os << "detection " << d.detection_id << " @ frame " << d.frame_number
     << ": conf " << d.confidence;
  return os;
}

//
// Generate some sample detections.
//

vector< user_detection_t >
make_sample_detections()
{
  return {
    { 100, 4, 0.3 },
    { 101, 4, 0.8 },
    { 102, 5, 0.5 }
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

vector< user_detection_t >
read_detections_from_stream( std::istream& is )
{
  namespace KPFC = KPF::canonical;

  vector< user_detection_t > dets;
  user_detection_t buffer;
  KPF::kpf_text_parser_t parser( is );
  KPF::kpf_reader_t reader( parser );

  while (reader
         >> KPF::reader< KPFC::id_t >( buffer.detection_id, KPFC::id_t::DETECTION_ID )
         >> KPF::reader< KPFC::timestamp_t>( buffer.frame_number, KPFC::timestamp_t::FRAME_NUMBER )
         >> KPF::reader< KPFC::conf_t>( buffer.confidence, DETECTOR_DOMAIN )
    )
  {
    dets.push_back( buffer );
    reader.flush();
  }
  return dets;
}

//
// Write a set of detections to a stream as KPF.
//

void
write_detections_to_stream( ostream& os,
                            const vector< user_detection_t >& dets )
{
  namespace KPFC = KPF::canonical;
  KPF::record_text_writer w( os );
  for (const auto& det: dets )
  {
    w
      << KPF::writer< KPFC::id_t >( det.detection_id, KPFC::id_t::DETECTION_ID )
      << KPF::writer< KPFC::timestamp_t >( det.frame_number, KPFC::timestamp_t::FRAME_NUMBER )
      << KPF::writer< KPFC::conf_t>( det.confidence, DETECTOR_DOMAIN )
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
  std::cout << "\nAbout to write detections:\n";
  write_detections_to_stream( ss, src_dets );
  std::cout << "KPF representation:\n" << ss.str();
  std::cout << "Done\n";

  std::cout << "\nAbout to read KPF:\n";
  vector< user_detection_t> new_dets = read_detections_from_stream( ss );
  for (auto i=0; i<new_dets.size(); ++i)
  {
    std::cout << "Converted det " << i << ": " << new_dets[i] << "\n";
  }
}
