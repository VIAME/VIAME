// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for draw_detected_object_set
 */

#ifndef ARROWS_OCV_DRAW_DETECTED_OBJECT_SET_H
#define ARROWS_OCV_DRAW_DETECTED_OBJECT_SET_H

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/draw_detected_object_set.h>

namespace kwiver {
namespace arrows {
namespace ocv {

/// An abstract base class for algorithms which draw tracks on top of
/// images in various ways, for analyzing results.
class KWIVER_ALGO_OCV_EXPORT draw_detected_object_set
  : public vital::algo::draw_detected_object_set
{
public:
  PLUGIN_INFO( "ocv",
               "Draw bounding box around detected objects on supplied image." )

  draw_detected_object_set();
  virtual ~draw_detected_object_set();

  virtual vital::config_block_sptr get_configuration() const;
  virtual void set_configuration(vital::config_block_sptr config);
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Draw detected object boxes om image.
  /**
   *
   * @param detected_set Set of detected objects
   * @param image Boxes are drawn in this image
   *
   * @return Image with boxes and other annotations added.
   */
  virtual kwiver::vital::image_container_sptr
    draw( kwiver::vital::detected_object_set_sptr detected_set,
          kwiver::vital::image_container_sptr image );

private:
  class priv;
  const std::unique_ptr<priv> d;
};

/// A smart pointer to a draw_tracks instance.
typedef std::shared_ptr<draw_detected_object_set> draw_detected_object_set_sptr;

} } } // end namespace

#endif // ARROWS_OCV_DRAW_DETECTED_OBJECT_SET_H
