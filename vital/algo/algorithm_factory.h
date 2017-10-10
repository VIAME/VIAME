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


#ifndef VITAL_ALGO_ALGORITHM_FACTORY_H
#define VITAL_ALGO_ALGORITHM_FACTORY_H

#include <vital/algo/vital_algo_export.h>
#include <vital/algo/algorithm.h>

#include <vital/vital_config.h>

#include <vital/plugin_loader/plugin_manager.h>

namespace kwiver {
namespace vital{

/**
 * @brief Factory class for algorithms.
 *
 * @tparam T Type of the object to be created.
 */
class VITAL_ALGO_EXPORT algorithm_factory
: public kwiver::vital::plugin_factory
{
public:
  /**
   * @brief CTOR for factory object
   *
   * This CTOR also takes a factory function so it can support
   * creating processes and clusters.
   *
   * @param algo Name of the algorithm
   * @param impl Name of the implementation
   */
  algorithm_factory( const std::string& algo,
                     const std::string& impl)
    : plugin_factory( algo ) // interface type
  {
    this->add_attribute( PLUGIN_NAME, impl )
      .add_attribute( PLUGIN_CATEGORY, "algorithm" );
  }

  virtual ~algorithm_factory() = default;

  algorithm_sptr create_object()
  {
    // Delegate to derived class
    return create_object_a();
    //+ could push algo and impl strings into the base algorithm class.
    //+ possibly through the CTOR
  }

private:
  virtual algorithm_sptr create_object_a() = 0;

};

  typedef std::shared_ptr< algorithm_factory >         algorithm_factory_handle_t;

// -----------------------------------------------------------------
template <class IMPL>
class algorithm_factory_0
  : public algorithm_factory
{
public:
  algorithm_factory_0( const std::string& algo,
                       const std::string& impl)
    : algorithm_factory( algo, impl)
  {
    this->add_attribute( CONCRETE_TYPE, typeid( IMPL ).name() );
  }

  virtual ~algorithm_factory_0() = default;

protected:
  virtual algorithm_sptr create_object_a()
  {
    algorithm_sptr new_obj = algorithm_sptr( new IMPL() );
    return new_obj;
  }
}; // end class algorithm_factory_0

// ------------------------------------------------------------------
/**
 * @brief Create algorithm from interface name and implementation name.
 *
 * @param algo_name Name of the interface
 * @param impl_name Name if the implementation
 *
 * @return New algorithm object or
 */
VITAL_ALGO_EXPORT
algorithm_sptr  create_algorithm( std::string const& algo_name,
                                  std::string const& impl_name );

/**
 * @brief Check the given type and implementation names against registered algorithms.
 *
 * \param type_name Type name of algorithm to validate
 * \param impl_name Implementation name of algorithm to validate
 * \returns true if the given \c type_name and \c impl_name describe a valid
 *          registered algorithm, or false if not.
 */
VITAL_ALGO_EXPORT
bool has_algorithm_impl_name(std::string const& type_name,
                             std::string const& impl_name);

/**
 * @brief Add an algorithm factory
 *
 * @param algo_name Algorithm name or interface type name
 * @param impl_name Implementation name or just name of plugin
 * @param conc_T Type of object to create
 *
 * @return
 */
#define ADD_ALGORITHM( impl_name, conc_T)                    \
  add_factory( new kwiver::vital::algorithm_factory_0<conc_T>( conc_T::static_type_name(), impl_name ))

} } // end namespace

#endif // VITAL_ALGO_ALGORITHM_FACTORY_H
