// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::activity_type C interface implementation
 */

#include "activity_type.h"

#include <vital/vital_config.h>
#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/activity_type.h>

#include <cstring>

namespace kwiver {
namespace vital_c {

// Allocate our shared pointer cache object
SharedPointerCache< kwiver::vital::activity_type, vital_activity_type_t >
AT_SPTR_CACHE( "activity_type" );

} }

// ------------------------------------------------------------------
vital_activity_type_t* vital_activity_type_new()
{
  STANDARD_CATCH(
    "C::activity_type::new", 0,
    auto cm_sptr = std::make_shared< kwiver::vital::activity_type > ();

    kwiver::vital_c::AT_SPTR_CACHE.store( cm_sptr );
    return reinterpret_cast<vital_activity_type_t*>( cm_sptr.get() );
    );
  return 0;
}

// ------------------------------------------------------------------
void vital_activity_type_destroy(vital_activity_type_t* obj)
{
  STANDARD_CATCH(
    "C::activity_type:::destroy", 0,
    kwiver::vital_c::AT_SPTR_CACHE.erase( obj );

    );
}

// ------------------------------------------------------------------
vital_activity_type_t*
vital_activity_type_new_from_list( VITAL_UNUSED vital_activity_type_t* obj,
                                   size_t count,
                                   char** class_names,
                                   VITAL_UNUSED double* scrs )
{
  STANDARD_CATCH(
    "C::activity_type::new_from_list", 0,
    std::vector<std::string> names;
    std::vector< double > scores;
    for (size_t i = 0; i < count; ++i)
    {
      names.push_back(class_names[i]);
      scores.push_back(scores[i]);
    }
    auto cm_sptr = std::make_shared< kwiver::vital::activity_type > ( names, scores );
    kwiver::vital_c::AT_SPTR_CACHE.store( cm_sptr );
    return reinterpret_cast<vital_activity_type_t*>( cm_sptr.get() );
    );
  return 0;
}

// ------------------------------------------------------------------
bool
vital_activity_type_has_class_name( vital_activity_type_t* obj, char* class_name )
{
  STANDARD_CATCH(
    "C::activity_type::has_class_name", 0,
    return kwiver::vital_c::AT_SPTR_CACHE.get( obj )->has_class_name( std::string( class_name ));
    );
  return false;
}

// ------------------------------------------------------------------
double vital_activity_type_score( vital_activity_type_t* obj, char* class_name )
{
  STANDARD_CATCH(
    "C::activity_type::score", 0,
    return kwiver::vital_c::AT_SPTR_CACHE.get( obj )->score( std::string( class_name ));
    );
  return 0;
}

// ------------------------------------------------------------------
char* vital_activity_type_get_most_likely_class( vital_activity_type_t* obj )
{
  STANDARD_CATCH(
    "C::activity_type::get_most_likely_class", 0,

    std::string class_name;
    double score;
    kwiver::vital_c::AT_SPTR_CACHE.get( obj )->get_most_likely( class_name, score );

    return strdup( class_name.c_str() );
    );
  return 0;
}

// ------------------------------------------------------------------
double vital_activity_type_get_most_likely_score( vital_activity_type_t* obj )
{
  STANDARD_CATCH(
    "C::activity_type::get_most_likely_score", 0,

    std::string class_name;
    double score;
    kwiver::vital_c::AT_SPTR_CACHE.get( obj )->get_most_likely( class_name, score );

    return score;
    );
  return 0;
}

// ------------------------------------------------------------------
void vital_activity_type_set_score( vital_activity_type_t* obj,
                                    char* class_name,
                                    double score )
{
  STANDARD_CATCH(
    "C::activity_type::set_score", 0,
    kwiver::vital_c::AT_SPTR_CACHE.get( obj )->set_score( std::string( class_name ), score);
    );
}

// ------------------------------------------------------------------
void vital_activity_type_delete_score( vital_activity_type_t* obj,
                                       char* class_name)
{
  STANDARD_CATCH(
    "C::activity_type::delete_score", 0,
    kwiver::vital_c::AT_SPTR_CACHE.get( obj )->delete_score( std::string( class_name ) );
    );
}

// ------------------------------------------------------------------
char** vital_activity_type_class_names( vital_activity_type_t* obj,
                                        VITAL_UNUSED double thresh )
{
  STANDARD_CATCH(
    "C::activity_type::class_names", 0,

    auto name_vector = kwiver::vital_c::AT_SPTR_CACHE.get( obj )->class_names();
    char** name_list = (char **) calloc( sizeof( char *), name_vector.size() +1 );

    for ( size_t i = 0; i < name_vector.size(); ++i )
    {
      name_list[i] = strdup( name_vector[i].c_str() );
    }

    return name_list;
    );
  return 0;
}

// ------------------------------------------------------------------
char** vital_activity_type_all_class_names( VITAL_UNUSED vital_activity_type_t* obj )
{
  STANDARD_CATCH(
    "C::activity_type::all_class_names", 0,

    auto name_vector = kwiver::vital::activity_type::all_class_names();
    char** name_list = (char **) calloc( sizeof( char *), name_vector.size() +1 );

    for ( size_t i = 0; i < name_vector.size(); ++i )
    {
      name_list[i] = strdup( name_vector[i].c_str() );
    }

    return name_list;
    );
  return 0;
}
