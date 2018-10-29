/*ckwg +29
 * Copyright 2015-2018 by Kitware, Inc.
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
 * \brief Common C Interface Utilities
 *
 * These utilities should only be used in CXX implementation files due to
 * their use of C++ structures.
 */

#ifndef VITAL_C_HELPERS_C_UTILS_H_
#define VITAL_C_HELPERS_C_UTILS_H_

#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <vital/bindings/c/error_handle.h>
#include <vital/exceptions/base.h>
#include <vital/logger/logger.h>

// There may be a better way to allocate this other than static CTOR
static auto m_logger( kwiver::vital::get_logger( "vital.c_utils" ) );

/// Macro allowing simpler population of an error handle
/**
 * Only does anything if error handle pointer is non-NULL.
 * \p msg should be a C string (char const*)
 *
 * If the given error handle has an existing message pointer, it will be freed
 * before setting the new message. If it is desired to retain the message for
 * some other purpose, it should be copied/duplicated before re-using an error
 * handle.
 *
 * \param eh_ptr Pointer to an error handle structure. May be null.
 * \param ec Integer error code.
 * \param msg C-string message to encode.
 */
#define POPULATE_EH(eh_ptr, ec, msg)                                                \
  do                                                                                \
  {                                                                                 \
    vital_error_handle_t *PEH_eh_ptr_cast =                                         \
        reinterpret_cast<vital_error_handle_t*>(eh_ptr);                            \
    if( PEH_eh_ptr_cast != NULL )                                                   \
    {                                                                               \
      PEH_eh_ptr_cast->error_code = ec;                                             \
      free(PEH_eh_ptr_cast->message); /* Does nothing if already null */            \
      /* +1 for null terminator */                                                  \
      PEH_eh_ptr_cast->message = (char*)malloc(sizeof(char) * (strlen(msg) + 1) );  \
      strcpy(PEH_eh_ptr_cast->message, msg);                                        \
    }                                                                               \
  } while(0)


/// Standardized try/catch for general use.
/**
 * If the provided code block contains a return, make sure to provide a
 * default/failure return after the use of the STANDARD_CATCH macro in case
 * an exception is thrown within the provided code block.
 *
 * Assuming \c eh_ptr points to an initialized vital_error_handle_t instance.
 * An arbitrary catch sets a -1 error code and assigns to the message field
 * the same thing that is printed to logging statement.
 */
#define STANDARD_CATCH(log_prefix, eh_ptr, code)                \
    do                                                          \
    {                                                           \
      try                                                       \
      {                                                         \
        code                                                    \
      }                                                         \
      catch( std::exception const &e )                          \
      {                                                         \
        std::ostringstream ss;                                  \
        ss << "Caught exception in C interface: " << e.what();  \
        POPULATE_EH( eh_ptr, -1, ss.str().c_str() );            \
      }                                                         \
      catch( char const* e )                                    \
      {                                                         \
        std::ostringstream ss;                                  \
        ss << "Caught error message: " << e;                    \
        POPULATE_EH( eh_ptr, -1, ss.str().c_str() );            \
      }                                                         \
      catch(...)                                                \
      {                                                         \
        std::string msg("Caught other exception");              \
        POPULATE_EH( eh_ptr, -1, msg.c_str() );                 \
      }                                                         \
    } while( 0 )


/// Wrap an optional string parameter.
/**
 * This converts a potentially null character array pointer ("string") to an
 * empty string literal (if it is null) or itself (otherwise). This is meant to
 * wrap <code>char const*</code> arguments that will be passed to C++ functions
 * taking \c std::string, as the latter does not allow construction from a null
 * pointer.
 */
#define MAYBE_EMPTY_STRING(s) (s?s:"")


/// Convenience macro for reinterpret cast pointer to a different type
/**
 * Most commonly used for conveniently converting C opaque pointer types into
 * their concrete C++ type when not shared_ptr controlled. We check that the
 * reinterpret cast yielded a non-null pointer.
 *
 * \param new_type The new type to reinterp cast \c ptr to. This should not
 *                 include the "*" as that is added in the macro.
 * \param ptr The pointer to convert.
 * \param var The variable to define in this macro. This should also be devoid
 *            of the "*" (controlled by macro).
 */
#define REINTERP_TYPE( new_type, ptr, var )             \
  new_type *var = reinterpret_cast< new_type* >( ptr ); \
  do                                                    \
  {                                                     \
    if( var == 0 )                                      \
    {                                                   \
      throw "Failed reinterpret cast";                  \
    }                                                   \
  } while(0)


/**
 * Convenience macro for dynamic casting a pointer to a different type with
 * error checking. This macro expects a {} block after its invocations that is
 * executed if the dynamic cast resulted in a NULL pointer (cast failure)
 *
 * \param new_type The new type to dynamic cast \c ptr to. This should not
 *                 include the "*" as that is controlled by the macro.
 * \param ptr The pointer to convert
 * \param var The variable to define in the macro. This should also be devoid of
 *            the "*" (controlled by macro).
 */
#define TRY_DYNAMIC_CAST( new_type, ptr, var )      \
  new_type *var = dynamic_cast< new_type* >( ptr ); \
  if( var == NULL )

namespace kwiver {
namespace vital_c {


/// Common shared pointer cache object
template < typename vital_t,  // VITAL type
           typename C_t >     // C Interface opaque type
class SharedPointerCache
{
public:
  typedef std::shared_ptr< vital_t > sptr_t;
  typedef std::map< vital_t const *, sptr_t > cache_t;
  typedef std::map< vital_t const *, size_t > ref_count_cache_t;

  /// Exception for when a given entry doesn't exist in this cache
  class NoEntryException
    : public kwiver::vital::vital_exception
  {
  public:
    NoEntryException( std::string const &reason )
    {
      this->m_what = reason;
    }
  };

  /// Exception for when we're asked to do something with a null pointer
  class NullPointerException
    : public kwiver::vital::vital_exception
  {
  public:
    NullPointerException( std::string const &reason)
    {
      this->m_what = reason;
    }
  };

  // ------------------------------------------------------------------
  /// Constructor
  SharedPointerCache( std::string name )
    : cache_(),
      ref_count_cache_(),
      name_( name )
  {}

  // ------------------------------------------------------------------
  /// Destructor
  virtual ~SharedPointerCache() = default;

  // ------------------------------------------------------------------
  /// Store a shared pointer
  void store( sptr_t sptr )
  {
    if( sptr.get() == NULL )
    {
      std::ostringstream ss;
      ss << get_log_prefix(sptr.get()) << ": Cannot store NULL pointer";
      throw NullPointerException(ss.str());
    }

    // If an sptr referencing the underlying pointer already exists in the map,
    // don't bother bashing the existing entry
    if( cache_.count( sptr.get() ) == 0 )
    {
      cache_[sptr.get()] = sptr;
      ref_count_cache_[sptr.get()] = 1;
    }
    else
    {
      ++ref_count_cache_[sptr.get()];
    }
  }

  // ------------------------------------------------------------------
  /// Access a stored shared pointer based on a supplied pointer
  sptr_t get( vital_t const *ptr ) const
  {
    if( ptr == NULL )
    {
      std::ostringstream ss;
      ss << get_log_prefix(ptr) << ": Cannot get NULL pointer";
      throw NullPointerException(ss.str());
    }

    typename cache_t::const_iterator it = cache_.find( ptr );
    if( it != cache_.end() )
    {
      return it->second;
    }
    else
    {
      std::ostringstream ss;
      ss << get_log_prefix(ptr) << ": "
         << " No cached shared_ptr for the given pointer (ptr: " << ptr << ")";
      throw NoEntryException( ss.str() );
    }
  }

  // ------------------------------------------------------------------
  /// Access a stored shared pointer based on the C interface opaque type
  sptr_t get( C_t const *ptr ) const
  {
    return this->get( reinterpret_cast< vital_t const * >( ptr ) );
  }

  // ------------------------------------------------------------------
  /// Erase an entry in the cache by vital-type pointer
  void erase( vital_t const *ptr )
  {
    if( ptr == NULL )
    {
      std::ostringstream ss;
      ss << get_log_prefix(ptr) << ": Cannot erase NULL pointer";
      throw NullPointerException(ss.str());
    }

    typename cache_t::iterator c_it = cache_.find( ptr );
    if( c_it != cache_.end() )
    {
      --ref_count_cache_[ptr];
      // Only finally erase cache entry when store references reaches 0
      if( ref_count_cache_[ptr] <= 0 )
      {
        cache_.erase(c_it);
        ref_count_cache_.erase(ptr);
      }
    }
  }

  // ------------------------------------------------------------------
  /// Erase an entry in the cache by C Interface opaque type pointer
  void erase( C_t const *ptr )
  {
    return this->erase( reinterpret_cast< vital_t const * >( ptr ) );
  }

private:
  /// Cache of shared pointers for concrete instances
  cache_t cache_;
  /// Number of times an instance has been "stored" in this cache
  /**
   * This is basically cache local reference counting ensuring that the cache
   * only actually erases a caches sptr when the number erase calls equals the
   * number of store calls for a given instance pointer.
   */
  ref_count_cache_t ref_count_cache_;
  /// Name of cache
  std::string name_;

  /// Helper method to generate logging prefix string
  std::string get_log_prefix( vital_t const *ptr ) const
  {
    std::ostringstream ss;
    ss << "SharedPointerCache::" << this->name_ << "::" << ptr;
    return ss.str();
  }
};


/// Helper function to create a char** list of strings given a vector of strings
void make_string_list( std::vector<std::string> const &list,
                       unsigned int &length, char ** &strings );

} } //end vital_c namespace

#endif //VITAL_C_HELPERS_C_UTILS_H_
