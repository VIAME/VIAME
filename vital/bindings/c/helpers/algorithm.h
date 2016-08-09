/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * \brief C++ Helper utilities for C interface of VITAL algorithms
 *
 * Private header for use in cxx implementation files.
 */

#ifndef VITAL_C_HELPERS_ALGORITHM_H_
#define VITAL_C_HELPERS_ALGORITHM_H_

#include <string>

#include <vital/bindings/c/helpers/c_utils.h>

#include <vital/algo/algorithm.h>
#include <vital/exceptions/algorithm.h>

#include <vital/bindings/c/algorithm.h>
#include <vital/bindings/c/helpers/config_block.h>


namespace kwiver {
namespace vital_c {

// Defined in ../c/algorithm.cxx
extern SharedPointerCache< kwiver::vital::algorithm,
                         vital_algorithm_t > ALGORITHM_SPTR_CACHE;


class invalid_algorithm_pointer
  : public kwiver::vital::vital_core_base_exception
{
public:
  invalid_algorithm_pointer( std::string reason )
  {
    this->m_what = reason;
  }
};

} } // end namespace


/// Macro companion to DECLARE_COMMON_ALGO_API, providing type implementations
#define DEFINE_COMMON_ALGO_API( type )                                  \
  /* Make sptr cache for specific type */                               \
  namespace kwiver {                                                    \
    namespace vital_c {                                                 \
      SharedPointerCache< kwiver::vital::algo::type, vital_algorithm_t > \
        ALGORITHM_##type##_SPTR_CACHE( #type );                         \
   } }                                                                     \
  /* ==================================================================== */ \
  /* Functions on types (static methods)                                  */ \
  /* -------------------------------------------------------------------- */ \
  /* Create new instance of a specific algorithm implementation */      \
  vital_algorithm_t* vital_algorithm_##type##_create( char const *impl_name ) \
  {                                                                     \
    STANDARD_CATCH(                                                     \
      "C::algorithm::" #type "::create", NULL,                          \
      kwiver::vital::algo::type##_sptr algo_sptr =                      \
        kwiver::vital::algo::type::create( impl_name );                 \
      if( algo_sptr )                                                   \
      {                                                                 \
        kwiver::vital_c::ALGORITHM_SPTR_CACHE.store( algo_sptr );       \
        kwiver::vital_c::ALGORITHM_##type##_SPTR_CACHE.store( algo_sptr ); \
        return reinterpret_cast<vital_algorithm_t*>(algo_sptr.get());   \
      }                                                                 \
    );                                                                  \
    return 0;                                                           \
  }                                                                     \
  /* Destroy an algorithm instance of this type */                      \
  void vital_algorithm_##type##_destroy( vital_algorithm_t *algo,       \
                                         vital_error_handle_t *eh )     \
  {                                                                     \
    STANDARD_CATCH(                                                     \
      "C::algorithm::" #type "::destroy", eh,                           \
      kwiver::vital_c::ALGORITHM_SPTR_CACHE.erase( algo );              \
      kwiver::vital_c::ALGORITHM_##type##_SPTR_CACHE.erase( algo );     \
    );                                                                  \
  }                                                                     \
  /* Get a list of registered implementation names for this algorithm type */ \
  void vital_algorithm_##type##_registered_names( unsigned int *length, \
                                                  char ***names )       \
  {                                                                     \
    STANDARD_CATCH(                                                     \
      "C::algorithm::" #type "::registered_names", NULL,                \
      std::vector<std::string> name_list =                              \
        kwiver::vital::algo::type::registered_names();                  \
      kwiver::vital_c::make_string_list( name_list, *length, *names );  \
    );                                                                  \
  }                                                                     \
  /** Get the configuration for a named algorithm in the given config */ \
  void                                                                  \
  vital_algorithm_##type##_get_type_config( char const *name,           \
                                            vital_algorithm_t *algo,    \
                                            vital_config_block_t *cb,   \
                                            vital_error_handle_t *eh )  \
  {                                                                     \
    STANDARD_CATCH(                                                     \
      "C::algorithm::" #type "::get_type_config", eh,                   \
      /* Checking algo ptr in order to allow getting a raw config when given \
       * NULL                                                           \
       */                                                               \
      kwiver::vital::algo::type##_sptr algo_sptr;                       \
      if( algo )                                                        \
      {                                                                 \
        algo_sptr = kwiver::vital_c::ALGORITHM_##type##_SPTR_CACHE.get( algo ); \
      }                                                                 \
      else                                                              \
      {                                                                 \
        algo_sptr = kwiver::vital::algo::type##_sptr();                 \
      }                                                                 \
      kwiver::vital::algo::type::get_nested_algo_configuration(         \
        name,                                                           \
        kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb ),             \
        algo_sptr                                                       \
      );                                                                \
    );                                                                  \
  }                                                                     \
  /** Set algorithm properties based on a named configuration in the config */ \
  void                                                                  \
  vital_algorithm_##type##_set_type_config( char const *name,           \
                                            vital_config_block_t *cb,   \
                                            vital_algorithm_t **algo,   \
                                            vital_error_handle_t *eh )  \
  {                                                                     \
    STANDARD_CATCH(                                                     \
      "C::algorithm::" #type "::set_type_config", eh,                   \
      kwiver::vital::algo::type##_sptr algo_sptr;                       \
      if( *algo )                                                       \
      {                                                                 \
        algo_sptr = kwiver::vital_c::ALGORITHM_##type##_SPTR_CACHE.get( *algo ); \
      }                                                                 \
      kwiver::vital::algo::type *orig_ptr = algo_sptr.get();            \
      kwiver::vital::algo::type::set_nested_algo_configuration(         \
        name,                                                           \
        kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb ),             \
        algo_sptr                                                       \
      );                                                                \
      /* If underlying pointer changed, destroy the old instance and register \
       * the new one.                                                   \
       */                                                               \
      if( orig_ptr != algo_sptr.get() )                                 \
      {                                                                 \
        if( orig_ptr )                                                  \
        {                                                               \
          kwiver::vital_c::ALGORITHM_SPTR_CACHE.erase( orig_ptr );      \
          kwiver::vital_c::ALGORITHM_##type##_SPTR_CACHE.erase( orig_ptr ); \
        }                                                               \
        kwiver::vital_c::ALGORITHM_SPTR_CACHE.store( algo_sptr );       \
        kwiver::vital_c::ALGORITHM_##type##_SPTR_CACHE.store( algo_sptr ); \
        *algo = reinterpret_cast<vital_algorithm_t*>( algo_sptr.get() ); \
      }                                                                 \
    );                                                                  \
  }                                                                     \
  /** Check the configuration with respect to this algorithm type */    \
  bool                                                                  \
  vital_algorithm_##type##_check_type_config( char const *name,         \
                                              vital_config_block_t *cb, \
                                              vital_error_handle_t *eh ) \
  {                                                                     \
    STANDARD_CATCH(                                                     \
      "C::algorithm::" #type "::check_type_config", eh,                 \
      return kwiver::vital::algo::type::check_nested_algo_configuration( \
        name,                                                           \
        kwiver::vital_c::CONFIG_BLOCK_SPTR_CACHE.get( cb )              \
      );                                                                \
    );                                                                  \
    return false;                                                       \
  }                                                                     \
  /* ==================================================================== */ \
  /* Functions on algorithm instances                                     */ \
  /* -------------------------------------------------------------------- */ \
  /* Clone the given algorithm instance */                              \
  vital_algorithm_t* vital_algorithm_##type##_clone( vital_algorithm_t *algo, \
                                                     vital_error_handle_t *eh ) \
  {                                                                     \
    STANDARD_CATCH(                                                     \
      "C::algorithm::" #type "::clone", eh,                             \
      if( algo )                                                        \
      {                                                                 \
        kwiver::vital::algo::type##_sptr new_sptr =                     \
          std::dynamic_pointer_cast<kwiver::vital::algo::type>(       \
            kwiver::vital_c::ALGORITHM_##type##_SPTR_CACHE.get( algo )->clone()); \
        if( new_sptr )                                                  \
        {                                                               \
          kwiver::vital_c::ALGORITHM_SPTR_CACHE.store( new_sptr );      \
          kwiver::vital_c::ALGORITHM_##type##_SPTR_CACHE.store( new_sptr ); \
          return reinterpret_cast<vital_algorithm_t*>(new_sptr.get());  \
        }                                                               \
        else                                                            \
        {                                                               \
          throw "Failed to clone instance of type '" #type "'";         \
        }                                                               \
      }                                                                 \
    );                                                                  \
    return 0;                                                           \
  }

#endif // VITAL_C_HELPERS_ALGORITHM_H_
