//
// An example of how one uses KPF to read and write custom data structures.
// YAML version.
//

//
// This example is in XX parts.
// Part 1 defines the user detection type we'll be playing with.


#include <vital/kpf/kpf_reader.h>
#include <vital/kpf/kpf_yaml_parser.h>
#include <vital/kpf/kpf_canonical_io_adapter.h>
#include <vital/kpf/kpf_yaml_writer.h>
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
// -----------------------------
// -----------------------------
//
// PART ONE: The user type we'll be playing with
//
// -----------------------------
// -----------------------------
//


struct user_activity_t
{
  int id;
  unsigned start, stop;
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
// Generate some sample detections.
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

struct user_act_adapter_t: public KPF::kpf_act_adapter< user_activity_t >
{
  user_act_adapter_t():
    kpf_act_adapter< user_activity_t >(
      // reads the canonical activity "a" into the user_activity "u"
      []( const KPF::canonical::activity_t& a, user_activity_t& u ) {
        u.id = a.activity_id.d;
        u.name = a.activity_name;
        u.start = a.timespan[0].tsr.start;
        u.stop = a.timespan[0].tsr.stop;
        for (const auto& actor: a.actors)
        {
          u.actor_ids.push_back( actor.id.d );
        }
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
        a.activity_name = u.name;
        a.activity_id.d = u.id;
        a.activity_id_domain = DETECTOR_DOMAIN;

        ostringstream oss;
        oss << u.confidence;
        a.attributes.push_back( KPF::canonical::kv_t( "conf", oss.str() ));

        KPF::canonical::activity_t::scoped_tsr_t tsr;
        tsr.domain = KPF::canonical::timestamp_t::FRAME_NUMBER;
        tsr.tsr.start = u.start;
        tsr.tsr.stop = u.stop;
        a.timespan.push_back( tsr );

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
