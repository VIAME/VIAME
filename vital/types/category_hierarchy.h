// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for category_hierarchy class
 */

#ifndef VITAL_CATEGORY_HIERARCHY_H_
#define VITAL_CATEGORY_HIERARCHY_H_

#include <vital/vital_export.h>

#include <string>
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
 * of arbitrary types of categories. Each category can have any number of
 * optional 'parent' and 'child' categories (for example an 'atlantic sea
 * scallop' is a type of broader 'scallop' category).
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
   * An object is created without class_names.
   */
  category_hierarchy();

  /**
   * @brief Disable copy constructor.
   *
   * Typically smart pointers to this class should be passed around.
   */
  category_hierarchy( const category_hierarchy& other ) = delete;

  /**
   * @brief Create an object from the given file containing a
   * hierarchy definition.
   *
   * @param filename Filename of file containing hierarchy.
   */
  category_hierarchy( std::string filename );

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
   * @brief Add a new class ID
   *
   * Parent name can be an empty string if this category has no parent.
   *
   * @param class_name Class name.
   */
  void add_class( const label_t& class_name,
                  const label_t& parent_name = label_t(""),
                  const label_id_t id = label_id_t(-1) );

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
   * @brief Get the official class name for the given class name.
   *
   * This is only useful for hierarchies with multiple synonyms
   * for the same item. In this case, the category "super-ID"
   * for the synonym will be returned.
   *
   * @param class_name Class name to get official class name for.
   *
   * @return \b Class label name.
   */
  label_t get_class_name( const label_t& class_name ) const;

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
   * @brief Adds another synonym to some existing class
   *
   * If a class is specified which doesn't exist, an exception will be thrown.
   *
   * @param class_name Existing class name.
   * @param synonym_name Synonym for the same class.
   */
  void add_synonym( const label_t& class_name, const label_t& synonym_name );

  /**
   * @brief Get list of all class_names in this hierarchy.
   *
   * The returned list will be ordered first based on the numerical ID number,
   * followed by alphabetically for items with no ID number. This does not
   * include synonyms.
   *
   * @return Ordered list of class_names.
   */
  label_vec_t all_class_names() const;

  /**
   * @brief Get list of all class_names without any children.
   *
   * The returned list will be ordered first based on the numerical ID number,
   * followed by alphabetically for items with no ID number. This does not
   * include synonyms.
   *
   * @return Ordered list of class_names without children.
   */
  label_vec_t child_class_names() const;

  /**
   * @brief Get number of class names on this object.
   *
   * This method returns the number of class names that are in this
   * object type.
   *
   * @return Number of registered class names.
   */
  size_t size() const;

  /**
   * @brief Load a hierarchy from a file
   *
   * Throwns on invalid file.
   */
  void load_from_file( const std::string& filename );

private:

  class category
  {
  public:
    label_t category_name;
    label_id_t category_id;

    std::vector< label_t > synonyms;

    std::vector< category* > parents;
    std::vector< category* > children;

    category()
     : category_name(""),
       category_id(-1)
    {}
  };

  using category_sptr = std::shared_ptr< category >;
  using hierarchy_map_t = std::map< label_t, category_sptr >;

  using hierarchy_const_itr_t = hierarchy_map_t::const_iterator;

  hierarchy_map_t m_hierarchy;

  hierarchy_const_itr_t find( const label_t& lbl ) const;
  std::vector< category_sptr > sorted_categories() const;
};

// typedef for a category_hierarchy shared pointer
typedef std::shared_ptr< category_hierarchy > category_hierarchy_sptr;

} }

#endif
