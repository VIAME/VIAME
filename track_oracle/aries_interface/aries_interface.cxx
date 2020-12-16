// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "aries_interface.h"
#include <iostream>

#include <vital/logger/logger.h>

static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::map;
using std::ostringstream;
using std::string;

#ifdef VIBRANT_AVAILABLE
using vidtk::event_types;
#endif

namespace kwiver {
namespace track_oracle {

aries_interface_impl* aries_interface::p = 0;

const char*
aries_interface_exception
::what() const throw()
{
  return this->msg.c_str();
}

struct aries_interface_impl
{
  map< string, size_t > activity_to_index_map;
  map< size_t, string > index_to_activity_map;
  map< size_t, string > index_to_PVO_map;
#ifdef VIBRANT_AVAILABLE
  map< event_types::enum_types, string > kwe_index_to_activity_map;
#endif
  map< unsigned, string > vpd_index_to_activity_map;

  enum {NO_PROMOTION, PROMOTE_TO_P, PROMOTE_TO_V};
  map< size_t, int > activity_to_PV_promotion_map;

  void load_activity_index_maps();
  void load_activity_pvo_maps();
#ifdef VIBRANT_AVAILABLE
  void load_kwe_maps();
#endif
  void load_vpd_maps();

  aries_interface_impl()
  {
    load_activity_index_maps();
    load_activity_pvo_maps();
#ifdef VIBRANT_AVAILABLE
    load_kwe_maps();
#endif
    load_vpd_maps();
  }

  size_t internal_lookup( const string& s ) const
  {
    const map< string, size_t >::const_iterator probe =
      this->activity_to_index_map.find( s );
    if (probe == this->activity_to_index_map.end())
    {
      throw aries_interface_exception( s );
    }
    return probe->second;
  }

};

/*
  initialize the map of activity names to their positions
  in the VIRAT/ARIES/ICE ("system") classifier vector.

  Formerly derived from the KWClassifierKeys_ structure in
  aries-src/src/cpp/aries/sir/DescriptorOracle.h,
  now found in
  aries-src/src/cpp/aries/communications/des/Utility.cpp
*/

void
aries_interface_impl
::load_activity_index_maps()
{

  // After the commit which broke the symmetry between
  // MITRE's ground truth labels and the labels in Utility.cpp,
  // this get a little trickier.
  //
  // We need to update the strings, but we also want to accept
  // the old ones.  This implies that the activity->index map must
  // be hidden, because if we have duplicate entries, then you can't
  // iterate over this map.  You *can* iterate over the backmap,
  // which will always return the canonical values.
  //
  // In other words, we read (accept) old and new values, but write
  // (emit) only new values.
  //
  // first, load the activity->index map
  // with the canonical names frmo Utility.cpp:

  this->activity_to_index_map.clear();
  size_t index = 0;
  this->activity_to_index_map[ "NotScored" ] = index++;
  this->activity_to_index_map[ "PersonDigging" ] = index++;
  this->activity_to_index_map[ "PersonGesturing" ] = index++;
  this->activity_to_index_map[ "PersonRunning" ] = index++;
  this->activity_to_index_map[ "PersonWalking" ] = index++;
  this->activity_to_index_map[ "PersonStanding" ] = index++;
  this->activity_to_index_map[ "PersonCarrying" ] = index++;
  this->activity_to_index_map[ "PersonLoadingVehicle" ] = index++;
  this->activity_to_index_map[ "PersonUnloadingVehicle" ] = index++;
  this->activity_to_index_map[ "PersonOpeningTrunk" ] = index++;
  this->activity_to_index_map[ "PersonClosingTrunk" ] = index++;
  this->activity_to_index_map[ "PersonEnteringVehicle" ] = index++;
  this->activity_to_index_map[ "PersonExitingVehicle" ] = index++;
  this->activity_to_index_map[ "PersonEnteringFacility" ] = index++;
  this->activity_to_index_map[ "PersonExitingFacility" ] = index++;
  this->activity_to_index_map[ "VehicleAccelerating" ] = index++;
  this->activity_to_index_map[ "VehicleDecelerating" ] = index++;
  this->activity_to_index_map[ "VehicleTurning" ] = index++;
  this->activity_to_index_map[ "VehicleStopping" ] = index++;
  this->activity_to_index_map[ "VehicleUTurn" ] = index++;
  this->activity_to_index_map[ "VehicleMaintainingDistance" ] = index++;
  this->activity_to_index_map[ "PersonThrowing" ] = index++;
  this->activity_to_index_map[ "VehicleStarting" ] = index++;
  this->activity_to_index_map[ "PersonCarryingTogether" ] = index++;
  this->activity_to_index_map[ "PersonClimbingAtop" ] = index++;
  this->activity_to_index_map[ "PersonKicking" ] = index++;
  this->activity_to_index_map[ "PersonLayingWire" ] = index++;
  this->activity_to_index_map[ "PersonSitting" ] = index++;
  this->activity_to_index_map[ "PersonPushing" ] = index++;
  this->activity_to_index_map[ "PersonPulling" ] = index++;
  this->activity_to_index_map[ "VehicleEnteringFacility" ] = index++;
  this->activity_to_index_map[ "VehicleExitingFacility" ] = index++;
  this->activity_to_index_map[ "VehiclePassing" ] = index++;
  this->activity_to_index_map[ "VehicleLooping" ] = index++;
  this->activity_to_index_map[ "VehicleNoVehicle" ] = index++;
  this->activity_to_index_map[ "EnteringRegion" ] = index++;
  this->activity_to_index_map[ "ExitingRegion" ] = index++;
  this->activity_to_index_map[ "Tripwire" ] = index++;
  this->activity_to_index_map[ "PersonMoving" ] = index++;
  this->activity_to_index_map[ "VehicleMoving" ] = index++;

  // now load up the index->activity map, before loading the aliases
  this->index_to_activity_map.clear();
  for ( map<string, size_t>::const_iterator i = this->activity_to_index_map.begin();
        i != this->activity_to_index_map.end();
        ++i)
  {
    this->index_to_activity_map[ i->second ] = i->first;
  }

  // now load the aliases for the old activities

  try
  {
    this->activity_to_index_map[ "Not Scored" ] = this->internal_lookup( "NotScored" );
    this->activity_to_index_map[ "Digging" ] = this->internal_lookup( "PersonDigging" );
    this->activity_to_index_map[ "Gesturing" ] = this->internal_lookup( "PersonGesturing" );
    this->activity_to_index_map[ "Running" ] = this->internal_lookup( "PersonRunning" );
    this->activity_to_index_map[ "Walking" ] = this->internal_lookup( "PersonWalking" );
    this->activity_to_index_map[ "Standing" ] = this->internal_lookup( "PersonStanding" );
    this->activity_to_index_map[ "Carrying" ] = this->internal_lookup( "PersonCarrying" );
    this->activity_to_index_map[ "Loading a Vehicle" ] = this->internal_lookup( "PersonLoadingVehicle" );
    this->activity_to_index_map[ "Unloading a Vehicle" ] = this->internal_lookup( "PersonUnloadingVehicle" );
    this->activity_to_index_map[ "Opening a Trunk" ] = this->internal_lookup( "PersonOpeningTrunk" );
    this->activity_to_index_map[ "Closing a Trunk" ] = this->internal_lookup( "PersonClosingTrunk" );
    this->activity_to_index_map[ "Getting Into a Vehicle" ] = this->internal_lookup( "PersonEnteringVehicle" );
    this->activity_to_index_map[ "Getting Out of a Vehicle" ] = this->internal_lookup( "PersonExitingVehicle" );
    this->activity_to_index_map[ "Entering a Facility" ] = this->internal_lookup( "PersonEnteringFacility" );
    this->activity_to_index_map[ "Exiting a Facility" ] = this->internal_lookup( "PersonExitingFacility" );
    this->activity_to_index_map[ "Accelerating" ] = this->internal_lookup( "VehicleAccelerating" );
    this->activity_to_index_map[ "Decelerating" ] = this->internal_lookup( "VehicleDecelerating" );
    this->activity_to_index_map[ "Turning" ] = this->internal_lookup( "VehicleTurning" );
    this->activity_to_index_map[ "Stopping" ] = this->internal_lookup( "VehicleStopping" );
    this->activity_to_index_map[ "U-Turn" ] = this->internal_lookup( "VehicleUTurn" );
    this->activity_to_index_map[ "VEHICLE_MAINTAINING_DISTANCE" ] = this->internal_lookup( "VehicleMaintainingDistance" );
    this->activity_to_index_map[ "PERSON_THROWING" ] = this->internal_lookup( "PersonThrowing" );
    this->activity_to_index_map[ "Starting" ] = this->internal_lookup( "VehicleStarting" );
    this->activity_to_index_map[ "PERSON_CARRYING_TOGETHER" ] = this->internal_lookup( "PersonCarryingTogether" );
    this->activity_to_index_map[ "PERSON_CLIMBING_ATOP" ] = this->internal_lookup( "PersonClimbingAtop" );
    this->activity_to_index_map[ "PERSON_KICKING" ] = this->internal_lookup( "PersonKicking" );
    this->activity_to_index_map[ "PERSON_LAYING_WIRE" ] = this->internal_lookup( "PersonLayingWire" );
    this->activity_to_index_map[ "PERSON_SITTING" ] = this->internal_lookup( "PersonSitting" );
    this->activity_to_index_map[ "PERSON_PUSHING" ] = this->internal_lookup( "PersonPushing" );
    this->activity_to_index_map[ "PERSON_PULLING" ] = this->internal_lookup( "PersonPulling" );
    this->activity_to_index_map[ "VEHICLE_DRIVING_INTO_A_FACILITY" ] = this->internal_lookup( "VehicleEnteringFacility" );
    this->activity_to_index_map[ "VEHICLE_DRIVING_OUT_OF_A_FACILITY" ] = this->internal_lookup( "VehicleExitingFacility" );
    this->activity_to_index_map[ "VEHICLE_PASSING" ] = this->internal_lookup( "VehiclePassing" );
    this->activity_to_index_map[ "VEHICLE_LOOPING" ] = this->internal_lookup( "VehicleLooping" );
    this->activity_to_index_map[ "VEHICLE_NO_VEHICLE" ] = this->internal_lookup( "VehicleNoVehicle" );
    this->activity_to_index_map[ "ENTERING_REGION" ] = this->internal_lookup( "EnteringRegion" );
    this->activity_to_index_map[ "EXITING_REGION" ] = this->internal_lookup( "ExitingRegion" );
    this->activity_to_index_map[ "TRIPWIRE" ] = this->internal_lookup( "Tripwire" );
    this->activity_to_index_map[ "PERSON_MOVING" ] = this->internal_lookup( "PersonMoving" );
    this->activity_to_index_map[ "VEHICLE_MOVING" ] = this->internal_lookup( "VehicleMoving" );

    // load up the activity->index to promotion map
    const char *promote_to_person[] = {"PersonRunning", "PersonWalking", "PersonCarrying",
                                       "PersonLoadingVehicle", "PersonUnloadingVehicle",
                                       "PersonOpeningTrunk", "PersonClosingTrunk",
                                       "PersonEnteringVehicle", "PersonExitingVehicle",
                                       "PersonEnteringFacility", "PersonExitingFacility",
                                       "PersonThrowing", "PersonClimbingAtop", "PersonKicking",
                                       "PersonLayingWire", "PersonPushing", "PersonPulling",
                                       "PersonCarryingTogether", 0};
    const char *promote_to_vehicle[] = {"VehicleAccelerating", "VehicleDecelerating",
                                        "VehicleTurning", "VehicleStopping", "VehicleUTurn",
                                        "VehicleMaintainingDistance", "VehicleStarting",
                                        "VehicleEnteringFacility", "VehicleExitingFacility",
                                        "VehiclePassing", "VehicleLooping", 0};

    for (unsigned i=0; promote_to_person[i] != 0; ++i)
    {
      size_t idx = this->internal_lookup( promote_to_person[i] );
      this->activity_to_PV_promotion_map[ idx ] = PROMOTE_TO_P;
    }
    for (unsigned i=0; promote_to_vehicle[i] != 0; ++i)
    {
      size_t idx = this->internal_lookup( promote_to_vehicle[i] );
      this->activity_to_PV_promotion_map[ idx ] = PROMOTE_TO_V;
    }
  }
  catch ( aries_interface_exception& e )
  {
    LOG_INFO( main_logger, e.what() );
    throw;
  }

  // all done
}

void
aries_interface_impl
::load_activity_pvo_maps()
{
  this->index_to_PVO_map.clear();

  /*
  string other   = string("other"  ),
             person  = string("person" ),
             vehicle = string("vehicle"),
             null    = string("null"   );
  */

  // construct index-2-PVO
  size_t index = 0;
  this->index_to_PVO_map[ index++ ]/*[ "Not Scored" ]                       */= aries_interface::PVO_NULL;
  this->index_to_PVO_map[ index++ ]/*[ "Digging" ]                          */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Gesturing" ]                        */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Running" ]                          */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Walking" ]                          */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Standing" ]                         */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Carrying" ]                         */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Loading a Vehicle" ]                */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Unloading a Vehicle" ]              */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Opening a Trunk" ]                  */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Closing a Trunk" ]                  */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Getting Into a Vehicle" ]           */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Getting Out of a Vehicle" ]         */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Entering a Facility" ]              */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Exiting a Facility" ]               */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Accelerating" ]                     */= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "Decelerating" ]                     */= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "Turning" ]                          */= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "Stopping" ]                         */= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "U-Turn" ]                           */= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "VEHICLE_MAINTAINING_DISTANCE" ]     */= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "PERSON_THROWING" ]                  */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "Starting" ]                         */= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "PERSON_CARRYING_TOGETHER" ]         */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "PERSON_CLIMBING_ATOP" ]             */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "PERSON_KICKING" ]                   */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "PERSON_LAYING_WIRE" ]               */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "PERSON_SITTING" ]                   */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "PERSON_PUSHING" ]                   */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "PERSON_PULLING" ]                   */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "VEHICLE_DRIVING_INTO_A_FACILITY" ]  */= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "VEHICLE_DRIVING_OUT_OF_A_FACILITY" ]*/= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "VEHICLE_PASSING" ]                  */= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "VEHICLE_LOOPING" ]                  */= aries_interface::PVO_VEHICLE;
  this->index_to_PVO_map[ index++ ]/*[ "VEHICLE_NO_VEHICLE" ]               */= aries_interface::PVO_NULL;
  this->index_to_PVO_map[ index++ ]/*[ "ENTERING_REGION" ]                  */= aries_interface::PVO_NULL;
  this->index_to_PVO_map[ index++ ]/*[ "EXITING_REGION" ]                   */= aries_interface::PVO_NULL;
  this->index_to_PVO_map[ index++ ]/*[ "TRIPWIRE" ]                         */= aries_interface::PVO_NULL;
  this->index_to_PVO_map[ index++ ]/*[ "PERSON_MOVING" ]                    */= aries_interface::PVO_PERSON;
  this->index_to_PVO_map[ index++ ]/*[ "VEHICLE_MOVING" ]                   */= aries_interface::PVO_VEHICLE;

}

#ifdef VIBRANT_AVAILABLE
/*
What's the VIRAT string for a kwe index?
*/

void
aries_interface_impl
::load_kwe_maps()
{
  this->kwe_index_to_activity_map.clear();

  this->kwe_index_to_activity_map[ event_types::STOPPING_EVENT ] = "VehicleStopping";
  this->kwe_index_to_activity_map[ event_types::STARTING_EVENT ] = "VehicleStarting";
  this->kwe_index_to_activity_map[ event_types::TURNING_EVENT ] = "VehicleTurning";
  this->kwe_index_to_activity_map[ event_types::UTURNING_EVENT ] = "VehicleUTurn";
  this->kwe_index_to_activity_map[ event_types::FOLLOWING_EVENT ] = "VehicleMaintainingDistance";
  this->kwe_index_to_activity_map[ event_types::PASSING_EVENT ] = "VehiclePassing";
  this->kwe_index_to_activity_map[ event_types::ACCELERATING_EVENT ] = "VehicleAccelerating";
  this->kwe_index_to_activity_map[ event_types::DECELERATING_EVENT ] = "VehicleDecelerating";
  this->kwe_index_to_activity_map[ event_types::EXIT_BUILDING_EVENT ] = "PersonExitingFacility";
  this->kwe_index_to_activity_map[ event_types::ENTER_BUILDING_EVENT ] = "PersonEnteringFacility";
  this->kwe_index_to_activity_map[ event_types::EXIT_VEHICLE_EVENT ] = "PersonExitingVehicle";
  this->kwe_index_to_activity_map[ event_types::ENTER_VEHICLE_EVENT ] = "PersonEnteringVehicle";
  this->kwe_index_to_activity_map[ event_types::WALKING_EVENT ] = "PersonWalking";
  this->kwe_index_to_activity_map[ event_types::STANDING_EVENT ] = "PersonStanding";
  this->kwe_index_to_activity_map[ event_types::RUNNING_EVENT ] = "PersonRunning";
}
#endif

/*
What's the VIRAT string for a vpd index?
*/

void
aries_interface_impl
::load_vpd_maps()
{
  this->vpd_index_to_activity_map.clear();

  // data from other/tier1/datasets/virat_public_dataset/Release2.0/ground/docs/README_format_release2.txt

  this->vpd_index_to_activity_map[ 1 ] = "PersonLoadingVehicle";
  this->vpd_index_to_activity_map[ 2 ] = "PersonUnloadingVehicle";
  this->vpd_index_to_activity_map[ 3 ] = "PersonOpeningTrunk";
  this->vpd_index_to_activity_map[ 4 ] = "PersonClosingTrunk";
  this->vpd_index_to_activity_map[ 5 ] = "PersonEnteringVehicle";
  this->vpd_index_to_activity_map[ 6 ] = "PersonExitingVehicle";
  this->vpd_index_to_activity_map[ 7 ] = "PersonGesturing";
  this->vpd_index_to_activity_map[ 8 ] = "PersonDigging";
  this->vpd_index_to_activity_map[ 9 ] = "PersonCarrying";
  this->vpd_index_to_activity_map[ 10 ] = "PersonRunning";
  this->vpd_index_to_activity_map[ 11 ] = "PersonEnteringFacility";
  this->vpd_index_to_activity_map[ 12 ] = "PersonExitingFacility";

  // from here down, seems to have been post-release 2.0

  this->vpd_index_to_activity_map[ 13 ] = "VehicleAccelerating";
  this->vpd_index_to_activity_map[ 14 ] = "VehicleMoving";
  this->vpd_index_to_activity_map[ 15 ] = "PersonWalking";
  this->vpd_index_to_activity_map[ 16 ] = "PersonStanding";

  //
  // traverse the map to ensure no typos (alas, at run-time)
  //
  try
  {
    for ( map<unsigned, string>::const_iterator i = this->vpd_index_to_activity_map.begin();
          i != this->vpd_index_to_activity_map.end();
          ++i )
    {
      this->internal_lookup( i->second );
    }
  }
  catch ( aries_interface_exception& e )
  {
    LOG_ERROR( main_logger, "Typo in vpd->aries map: " << e.what() );
    throw;
  }
}

bool
aries_interface
::promote_to_PERSON_MOVING( size_t index )
{
  if (! p ) p = new aries_interface_impl();

  map< size_t, int >::const_iterator probe = p->activity_to_PV_promotion_map.find( index );
  return (probe != p->activity_to_PV_promotion_map.end()) &&
    (probe->second == aries_interface_impl::PROMOTE_TO_P);
}

bool
aries_interface
::promote_to_VEHICLE_MOVING( size_t index )
{
  if (! p ) p = new aries_interface_impl();

  map< size_t, int >::const_iterator probe = p->activity_to_PV_promotion_map.find( index );
  return (probe != p->activity_to_PV_promotion_map.end()) &&
    (probe->second == aries_interface_impl::PROMOTE_TO_V);
}

size_t
aries_interface
::activity_to_index( const string& s )
{
  if ( ! p ) p = new aries_interface_impl();

  return p->internal_lookup( s );
}

const map< size_t, string >&
aries_interface
::index_to_activity_map()
{
  if ( ! p ) p = new aries_interface_impl();

  return p->index_to_activity_map;
}

#ifdef VIBRANT_AVAILABLE
string
aries_interface
::kwe_index_to_activity( event_types::enum_types kwe_index )
{
  if ( ! p ) p = new aries_interface_impl();

  map< event_types::enum_types, string >::const_iterator probe =
    p->kwe_index_to_activity_map.find( kwe_index );
  return
    ( probe == p->kwe_index_to_activity_map.end() )
    ? ""
    : probe->second;
}
#endif

string
aries_interface
::vpd_index_to_activity( unsigned vpd_index )
{
  if ( ! p ) p = new aries_interface_impl();

  map< unsigned, string >::const_iterator probe =
    p->vpd_index_to_activity_map.find( vpd_index );
  return
    ( probe == p->vpd_index_to_activity_map.end() )
    ? ""
    : probe->second;
}

string
aries_interface
::activity_to_PVO( const string& s )
{
  if ( ! p ) p = new aries_interface_impl();

  size_t idx = p->internal_lookup( s );
  return p->index_to_PVO_map[ idx ];
}

const map< size_t, string >&
aries_interface
::index_to_PVO_map()
{
  if ( ! p ) p = new aries_interface_impl();

  return p->index_to_PVO_map;
}

const string
aries_interface
::PVO_PERSON = "person";

const string
aries_interface
::PVO_VEHICLE = "vehicle";

const string
aries_interface
::PVO_OTHER = "other";

const string
aries_interface
::PVO_NULL = "null";

} // ...track_oracle
} // ...kwiver
