/*ckwg +29
 * Copyright 2016-2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief vital::class_map C interface implementation
 */

#include "class_map.h"

#include <vital/types/class_map.h>

#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/helpers/class_map.h>

#include <cstring>

namespace kwiver {
namespace vital_c {

// Allocate our shared pointer cache object
SharedPointerCache< kwiver::vital::class_map, vital_class_map_t >
  CM_SPTR_CACHE( "class_map" );

} }


// ------------------------------------------------------------------
vital_class_map_t* vital_class_map_new()
{
  STANDARD_CATCH(
    "C::class_map:new", 0,
    auto cm_sptr = std::make_shared< kwiver::vital::class_map> ();

    kwiver::vital_c::CM_SPTR_CACHE.store( cm_sptr );
    return reinterpret_cast<vital_class_map_t*>( cm_sptr.get() );
  );
  return 0;
}


// ------------------------------------------------------------------
void vital_class_map_destroy(vital_class_map_t* obj)
{
  STANDARD_CATCH(
    "C::class_map::destroy", 0,
    kwiver::vital_c::CM_SPTR_CACHE.erase( obj );

  );
}


// ------------------------------------------------------------------
vital_class_map_t* vital_class_map_new_from_list( vital_class_map_t* obj,
                                                  size_t count,
                                                  char** class_names,
                                                  double* scores )
{
  STANDARD_CATCH(
    "C::class_map:new_from_list", 0,
    std::vector<std::string> names;
    std::vector< double > scores;
    for (size_t i = 0; i < count; ++i)
    {
      names.push_back(class_names[i]);
      scores.push_back(scores[i]);
    }
    auto cm_sptr = std::make_shared< kwiver::vital::class_map> ( names, scores );
    kwiver::vital_c::CM_SPTR_CACHE.store( cm_sptr );
    return reinterpret_cast<vital_class_map_t*>( cm_sptr.get() );
  );
  return 0;
}


// ------------------------------------------------------------------
bool vital_class_map_has_class_name( vital_class_map_t* obj, char* class_name )
{
  STANDARD_CATCH(
    "C::class_map:has_class_name", 0,
    return kwiver::vital_c::CM_SPTR_CACHE.get( obj )->has_class_name( std::string( class_name ));
    );
  return false;
}


// ------------------------------------------------------------------
double vital_class_map_score( vital_class_map_t* obj, char* class_name )
{
  STANDARD_CATCH(
    "C::class_map:score", 0,
    return kwiver::vital_c::CM_SPTR_CACHE.get( obj )->score( std::string( class_name ));
    );
  return 0;
}


// ------------------------------------------------------------------
char* vital_class_map_get_most_likely_class( vital_class_map_t* obj )
{
  STANDARD_CATCH(
    "C::class_map:get_most_likely_class", 0,

    std::string class_name;
    double score;
    kwiver::vital_c::CM_SPTR_CACHE.get( obj )->get_most_likely( class_name, score );

    return strdup( class_name.c_str() );
    );
  return 0;
}


// ------------------------------------------------------------------
double vital_class_map_get_most_likely_score( vital_class_map_t* obj )
{
  STANDARD_CATCH(
    "C::class_map:get_most_likely_score", 0,

    std::string class_name;
    double score;
    kwiver::vital_c::CM_SPTR_CACHE.get( obj )->get_most_likely( class_name, score );

    return score;
    );
  return 0;
}


// ------------------------------------------------------------------
void vital_class_map_set_score( vital_class_map_t* obj,
                                char* class_name,
                                double score )
{
  STANDARD_CATCH(
    "C::class_map:set_score", 0,
    kwiver::vital_c::CM_SPTR_CACHE.get( obj )->set_score( std::string( class_name ), score);
    );
}


// ------------------------------------------------------------------
void vital_class_map_delete_score( vital_class_map_t* obj,
                                   char* class_name)
{
  STANDARD_CATCH(
    "C::class_map:delete_score", 0,
    kwiver::vital_c::CM_SPTR_CACHE.get( obj )->delete_score( std::string( class_name ) );
    );
}


// ------------------------------------------------------------------
char** vital_class_map_class_names( vital_class_map_t* obj,
                                    double thresh )
{
  STANDARD_CATCH(
    "C::class_map:class_names", 0,

    auto name_vector = kwiver::vital_c::CM_SPTR_CACHE.get( obj )->class_names();
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
char** vital_class_map_all_class_names(vital_class_map_t* obj)
{
  STANDARD_CATCH(
    "C::class_map:all_class_names", 0,

    auto name_vector = kwiver::vital::class_map::all_class_names();
    char** name_list = (char **) calloc( sizeof( char *), name_vector.size() +1 );

    for ( size_t i = 0; i < name_vector.size(); ++i )
    {
      name_list[i] = strdup( name_vector[i].c_str() );
    }

    return name_list;
    );
  return 0;
}
