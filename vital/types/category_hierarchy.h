/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \brief Interface for category_hierarchy class
 */

#ifndef VITAL_CATEGORY_HIERARCHY_H_
#define VITAL_CATEGORY_HIERARCHY_H_

#include <vital/vital_export.h>

#include <vector>
#include <map>
#include <memory>

namespace kwiver {
namespace vital {


// -----------------------------------------------------------------------------
/**
 * @brief Category Hierarchy.
 *
 * A relatively simple class used for representing semantic hierarchies
 * of arbitrary types of categories.
 */
class VITAL_EXPORT category_hierarchy
{
public:

  using label_t = std::string;
  using label_id_t = int;

  using label_vec_t = std::vector< label_t >;
  using label_id_vec_t = std::vector< label_id_t >;

  /**
   * @brief Create an empty object.
   *
   * An object is created without class_names
   */
  category_hierarchy();

  /**
   * @brief Create a new categorical hierarchy class.
   *
   * Create a new category hierarchy instance with a set of labels. If
   * a class has no parent category, it can be left as an empty string.
   *
   * @param class_names List of labels for the possible classes.
   * @param parent_names Optional list of parent labels for all classes.
   * @param ids Optional list of numerical IDs for each class label.
   */
  category_hierarchy( const label_vec_t& class_names,
                      const label_vec_t& parent_names = label_vec_t(),
                      const label_id_vec_t& ids = label_id_vec_t() );

  /**
   * @brief Default deconstructor
   *
   * Destroy object, deallocating all utilized memory
   */
  ~category_hierarchy();

  /**
   * @brief Determine if class_name is present.
   *
   * This method determines if the specified class name is present in
   * this object.
   *
   * @param class_name Class name to test.
   *
   * @return \b true if class name is present.
   */
  bool has_class_name( const label_t& class_name ) const;

  /**
   * @brief Add a new class ID
   *
   * @param class_name Class name.
   */
  void add_class( const label_t& class_name,
                  const label_t& parent_name = label_t(),
                  const label_id_t id = label_id_t() );

  /**
   * @brief Get the class label ID for the given class name.
   *
   * @param class_name Class name to get ID for.
   *
   * @return \b Class label ID.
   */
  label_id_t get_class_id( const label_t& class_name ) const;

  /**
   * @brief Get the class parent labels for the given class name.
   *
   * @param class_name Class name to get parents for.
   *
   * @return \b Class parent label vector.
   */
  label_vec_t get_class_parents( const label_t& class_name ) const;

  /**
   * @brief Add a parent-child relationship between two classes
   *
   * If a class is specified which doesn't exist, an exception will be thrown.
   *
   * @param child_name Child name.
   * @param parent_name Parent name.
   */
  void add_relationship( const label_t& child_name, const label_t& parent_name );

  /**
   * @brief Get list of class_names for this hierarchy.
   *
   * @return Ordered list of class_names.
   */
  label_vec_t class_names() const;

  /**
   * @brief Get number of class names on this object.
   *
   * This method returns the number of class names that are in this
   * object type.
   *
   * @return Number of registered class names.
   */
  size_t size() const;

private:

  class category
  {
  public:
    label_t category_name;
    label_id_t category_id;

    std::vector< category* > parents;
    std::vector< category* > children;
  };

  std::map< label_t, category* > m_hierarchy;

  std::map< label_t, category* >::const_iterator find( const label_t& lbl ) const;
};

// typedef for a category_hierarchy shared pointer
typedef std::shared_ptr< category_hierarchy > category_hierarchy_sptr;

} }

#endif
