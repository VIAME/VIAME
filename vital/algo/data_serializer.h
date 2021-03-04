// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief data_serializer algorithm definition
 */

#ifndef VITAL_ALGO_SERIALIZE_H
#define VITAL_ALGO_SERIALIZE_H

#include <vital/vital_config.h>
#include <vital/algo/algorithm.h>
#include <vital/any.h>

#include <memory>
#include <string>
#include <set>
#include <map>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class serializing and deserializing.
/**
 * This class represents a pair of methods that serialize and
 * deserialize concrete data types. These methods are guaranteed to
 * work together. A data type serialized and deserialized by this
 * algorithm are semantically equivalent. The format and process of
 * actually doing the serialization depends on the concrete
 * implementation.
 *
 * It is expected that implementations of this interface will not
 * require any implementation specific configuration parameters. This
 * is because the implementation is selected at run time based on the
 * data type of the port connections.
 *
 * The serializer is stateless and idempotent.
 *
 * The main application for this algorithm is to serialize data
 * objects for inter-process communications.
 */
class VITAL_ALGO_EXPORT data_serializer
  : public kwiver::vital::algorithm_def< data_serializer >
{
public:

  /// Return the name of this algorithm
  static std::string static_type_name() { return "data_serializer"; }

  /// Serialize the item into a byte string.
  /**
   * This method serializes the supplied data item(s) and returns a
   * byte string.
   *
   * All implementations must define the set of expected element
   * names. An exception is thrown if a names is supplied and is not
   * known to the implementation. An implementation can elect to
   * accept an input map that does not contain all supported element
   * names.
   *
   * If the implementation only supports a single data element, then
   * the name associated with that element will be "datum".
   *
   * The type of the data represented by the datum must match the type
   * expected by the serializer implementation or an exception will be
   * thrown.
   *
   * @param elements Data items to be serialized.
   *
   * @return Byte string of serialized data item.
   *
   * @throws kwiver::vital::bad_any_cast
   * @throws kwiver::vital::serialization - for unexpected element name
   */
  virtual std::shared_ptr< std::string > serialize( const vital::any& element ) = 0;

  /// Deserialize byte string into data type.
  /**
   * Deserialize the supplied string of bytes into new data
   * item(s). This method must handle an input byte string created by the
   * \c serialize() method and convert it to the concrete type(s). The
   * actual type used for the conversion is based on the concrete
   * implementation of this algorithm. If the input byte string does
   * not represent the expected data type, then an exception will be
   * thrown.
   *
   * All implementations must define the set of expected element
   * names. An exception is thrown if a names is supplied and is not
   * known to the implementation. An implementation can elect to
   * accept an input map that does not contain all supported element
   * names.
   *
   * If the implementation only supports a single data element, then
   * the name associated with that element will be "datum".
   *
   * @param message Serialized data item that is to be processed.
   *
   * @return Concrete data type, represented as an any, created from
   * the input.
   *
   * @throws kwiver::vital::bad_any_cast
   * @throws kwiver::vital::serialization - for unexpected element name
   */
  virtual vital::any deserialize( const std::string& message ) = 0;

  virtual void set_configuration( kwiver::vital::config_block_sptr config ) { }
  virtual bool check_configuration(config_block_sptr config) const { return true; }

protected:
  data_serializer();

};

/// Shared pointer for detect_features algorithm definition class
typedef std::shared_ptr<data_serializer> data_serializer_sptr;

} // end namespaces
}
}

#endif // VITAL_ALGO_SERIALIZE_H
