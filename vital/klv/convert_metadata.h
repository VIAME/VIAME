// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file \brief This file contains the internal interface for
 * converter class.
 */

#ifndef KWIVER_VITAL_KLV_CONVERT_METADATA_H_
#define KWIVER_VITAL_KLV_CONVERT_METADATA_H_

#include <vital/klv/vital_klv_export.h>

#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>

#include <vital/klv/klv_0601.h>
#include <vital/klv/klv_0104.h>
#include <vital/klv/klv_parse.h>
#include <vital/util/any_converter.h>

#include <vital/logger/logger.h>

namespace kwiver {
namespace vital {

// ----------------------------------------------------------------
/**
 * @brief
 *
 */
class VITAL_KLV_EXPORT convert_metadata
{
public:
  // -- CONSTRUCTORS --
  convert_metadata();
  ~convert_metadata();

  /**
   * @brief Convert raw metadata packet into vital metadata entries.
   *
   * @param[in] klv Raw metadata packet containing UDS key
   * @param[in,out] metadata Collection of metadata this updated.
   *
   * @throws metadata_exception When error encountered.
   */
   void convert( klv_data const& klv, metadata& md );

  /**
   * \brief Get type representation for vital metadata tag.
   *
   * This method returns the type id string for the specified vital
   * metadata tag.
   *
   * \param tag Code for metadata tag.
   *
   * \return Type info for this tag
   */
  static std::type_info const& typeid_for_tag( vital_metadata_tag tag );

  /** Constants used to determine the source of this metadata
   * collection. The value of the VITAL_META_METADATA_ORIGIN tag is
   * set to one of the following values depending on the format of the
   * metadata packet processed.
   *
   * Typical usage is:
   \code
   std::string type;
   if (meta.has( VITAL_META_METADATA_ORIGIN ) )
   {
      type = meta.find( VITAL_META_METADATA_ORIGIN ).as_string();
   }
   if (metadata::MISB_0104 == type)
   {
       // metadata was from MISB 0104 packet
   }
   \endcode
   */
  const static std::string MISB_0104;
  const static std::string MISB_0601;

private:

  void convert_0601_metadata( klv_lds_vector_t const& lds, metadata& md );
  void convert_0104_metadata( klv_uds_vector_t const& uds, metadata& md );

  kwiver::vital::any normalize_0601_tag_data( klv_0601_tag tag,
                                              kwiver::vital::vital_metadata_tag vital_tag,
                                              kwiver::vital::any const& data );

  kwiver::vital::any normalize_0104_tag_data( klv_0104::tag tag,
                                            kwiver::vital::vital_metadata_tag vital_tag,
                                            kwiver::vital::any const& data );

  kwiver::vital::logger_handle_t m_logger;

  any_converter< double > convert_to_double;
  any_converter< uint64_t > convert_to_int;

  metadata_traits m_metadata_traits;

}; // end class convert_metadata

} } // end namespace

#endif
