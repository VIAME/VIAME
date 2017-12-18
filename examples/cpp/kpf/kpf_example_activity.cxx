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
 * \brief KPF read / write of the activity type, using YAML.
 *
 * This file is similar to kpf_example_complex.cxx, but demonstrates
 * reading and writing the KPF activity object.
 *
 */

#include <arrows/kpf/yaml/kpf_reader.h>
#include <arrows/kpf/yaml/kpf_yaml_parser.h>
#include <arrows/kpf/yaml/kpf_canonical_io_adapter.h>
#include <arrows/kpf/yaml/kpf_yaml_writer.h>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <cctype>

using std::string;
using std::vector;
using std::pair;
using std::make_pair;
using std::istringstream;
using std::stringstream;
using std::ostringstream;
using std::ostream;
using std::stod;

namespace KPF=kwiver::vital::kpf;


//
// Our user-defined activity object.
// Activities can have multiple actors.
//

struct user_activity_t
{
  int id;
  unsigned start, stop; // in frame numbers
  string name;
  vector<size_t> actor_ids;
  double confidence;
  user_activity_t()
    : id(0), start(0), stop(0), name("invalid"), confidence(-1.0)
  {}
  user_activity_t( int i, unsigned start_, unsigned stop_, const string& n,
                   const vector<size_t>& a, double c)
    : id(i), start(start_), stop(stop_), name(n), actor_ids(a), confidence(c)
  {}
};

//
// pretty-print
//

ostream& operator<<( ostream& os, const user_activity_t& a )
{
  os << "activity " << a.id << " is '" << a.name << "'; start/stop " << a.start << " / " << a.stop;
  os << "; actors: ";
  for (auto i: a.actor_ids )
  {
    os << i << ", ";
  }
  os << "; conf: " << a.confidence;
  return os;
}

//
// Generate some sample activities.
//

vector< user_activity_t >
make_sample_activities()
{
  return {
    { 100, 1001, 1010, "walking", {15}, 0.3},
    { 102, 1500, 1600, "crowding", {16,17,18}, 0.5},
    { 104, 0, 100, "juggling", {19,20,21,22}, 0.4}
  };
}


//
// This is the socially-agreed domain for the detector; an arbitrary number
// greater than 9 which disambiguates this detector from others we may
// have on the project.
//

const int DETECTOR_DOMAIN=17;
const int ACTOR_DOMAIN=15;

//
// The KPF activity object is complex, and requires an adapter.
//

struct user_act_adapter_t: public KPF::kpf_act_adapter< user_activity_t >
{
  user_act_adapter_t():
    kpf_act_adapter< user_activity_t >(
      // reads the canonical activity "a" into the user_activity "u"
      []( const KPF::canonical::activity_t& a, user_activity_t& u ) {
        // load the activity ID, name, and start and stop frames
        u.id = a.activity_id.d;
        u.name = a.activity_name;
        u.start = a.timespan[0].tsr.start;
        u.stop = a.timespan[0].tsr.stop;
        // load in our actor IDs
        for (const auto& actor: a.actors)
        {
          u.actor_ids.push_back( actor.id.d );
        }
        // look for our confidence value in the key/value pairs
        for (const auto& kv: a.attributes)
        {
          if (kv.key == "conf")
          {
            u.confidence = stod(kv.val);
          }
        }
      },

      // converts a user_activity "a" into a canonical activity and returns it
      []( const user_activity_t& u ) {
        KPF::canonical::activity_t a;
        // set the name, ID, and domain
        a.activity_name = u.name;
        a.activity_id.d = u.id;
        a.activity_id_domain = DETECTOR_DOMAIN;

        // set our confidence as a key/value pair
        ostringstream oss;
        oss << u.confidence;
        a.attributes.push_back( KPF::canonical::kv_t( "conf", oss.str() ));

        // set the start / stop time (as frame numbers)
        KPF::canonical::activity_t::scoped_tsr_t tsr;
        tsr.domain = KPF::canonical::timestamp_t::FRAME_NUMBER;
        tsr.tsr.start = u.start;
        tsr.tsr.stop = u.stop;
        a.timespan.push_back( tsr );

        // also use the activity start/stop time for each actor
        for (auto actor:u.actor_ids)
        {
          a.actors.push_back( {ACTOR_DOMAIN, KPF::canonical::id_t(actor), a.timespan });
        }

        return a;
      } )
  {}
};


//
// Read a set of activities from a stream.
//
// Note that we're implicitly expecting to find record one per line.
//

vector< user_activity_t >
read_activities_from_stream( std::istream& is )
{
  namespace KPFC = KPF::canonical;
  vector< user_activity_t > acts;
  user_act_adapter_t act;
  KPF::kpf_yaml_parser_t parser( is );
  KPF::kpf_reader_t reader( parser );
  user_activity_t buffer;

  while (reader
         >> KPF::reader< KPFC::activity_t >( act, DETECTOR_DOMAIN )
    )
  {
    act.get( buffer );
    acts.push_back( buffer );
    reader.flush();
  }

  return acts;
}

//
// Write a set of activities to a stream as KPF.
//

void
write_activities_to_stream( ostream& os,
                            const vector< user_activity_t >& acts )
{
  namespace KPFC = KPF::canonical;
  user_act_adapter_t act_adapter;
  KPF::record_yaml_writer w( os );
  for (const auto& act: acts )
  {
    w
      << KPF::writer< KPFC::activity_t >( act_adapter( act ), DETECTOR_DOMAIN )
      << KPF::record_yaml_writer::endl;
  }
}

int main()
{

  vector< user_activity_t > src = make_sample_activities();
  for (auto i=0; i<src.size(); ++i)
  {
    std::cout << "Source act " << i << ": " << src[i] << "\n";
  }

  stringstream ss;
  std::cout << "\nAbout to write activities:\n";
  write_activities_to_stream( ss, src );
  std::cout << "KPF representation:\n" << ss.str();
  std::cout << "Done\n";

  std::cout << "\nAbout to read KPF:\n";
  vector< user_activity_t> new_acts = read_activities_from_stream( ss );
  for (auto i=0; i<new_acts.size(); ++i)
  {
    std::cout << "Converted act " << i << ": " << new_acts[i] << "\n";
  }

}
