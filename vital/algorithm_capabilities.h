/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

#ifndef VITAL_ALGORITHM_CAPABILITIES_H
#define VITAL_ALGORITHM_CAPABILITIES_H

#include <vital/vital_export.h>

#include <string>
#include <vector>
#include <memory>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * \brief Algorithm capability set.
 *
 * This class represents a collection of capability for a concrete
 * algorithm.
 *
 * Capabilities provide a way to flexibly query the concrete
 * implementation and determine the capabilities, features and
 * limitations.
 *
 */
class VITAL_EXPORT algorithm_capabilities
{
public:
  typedef std::string capability_name_t;
  typedef std::vector< capability_name_t > capability_list_t;

  algorithm_capabilities();
  algorithm_capabilities( algorithm_capabilities const& other );
  ~algorithm_capabilities();


  /// Indicate if capability is supported.
  /**
   * This method reports if the specified capability is supported by the
   * concrete implementation. If the capability is supported, then the
   * value can be accessed with the capability() method. The value may be
   * \b true or \b false.
   *
   * \param name Capability name
   *
   * \return \b true if capability is supported, \b false otherwise.
   */
  bool has_capability( capability_name_t const& name ) const;


  /// Get list of supported capabilities.
  /**
   * This method returns a vector of all capabilities supported by the
   * current algorithm implementation. Only the names are returned.
   *
   * @return Vector of supported capabilities.
   */
  capability_list_t capability_list() const;


  /// Return value of capability,
  /**
   * This method returns the value of the specified capability.  \b false
   * is also returned if the capability does not exist.  it is a
   * best-practice to call has_capability() to determine if capability is
   * present before getting its value, since a \b false return is
   * otherwise ambiguous.
   *
   * @param name Capability name.
   *
   * @return Value of capability.
   */
  bool capability( capability_name_t const& name ) const;


  /// Set capability value.
  /**
   * This method creates a capability and sets it to the specified value.
   * The value is replaced if the capability already exists.
   *
   * @param name Capability name
   * @param val Capability value
   */
  void set_capability( capability_name_t const& name, bool val );

  algorithm_capabilities& operator=( algorithm_capabilities const& other );

private:
  /// private implementation class
  class priv;
  std::unique_ptr<priv> d;
};

} } // end namespace

#endif /* VITAL_ALGORITHM_CAPABILITIES_H */
