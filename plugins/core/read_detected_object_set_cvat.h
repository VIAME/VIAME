/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Interface for read_detected_object_set_cvat
 */

#ifndef VIAME_CORE_READ_DETECTED_OBJECT_SET_CVAT_H
#define VIAME_CORE_READ_DETECTED_OBJECT_SET_CVAT_H

#include "viame_core_export.h"

#include <vital/algo/detected_object_set_input.h>

#include <memory>

namespace viame {

/// \brief Read detected object sets from CVAT XML format.
///
/// CVAT (Computer Vision Annotation Tool) XML format contains:
///   - annotations root element
///   - meta section with task info and labels
///   - image elements with id, name, width, height attributes
///   - box elements with label, xtl, ytl, xbr, ybr attributes
///   - optionally polygon elements with label and points attributes
///
class VIAME_CORE_EXPORT read_detected_object_set_cvat
  : public kwiver::vital::algo::detected_object_set_input
{
public:
  static constexpr char const* name = "cvat";

  static constexpr char const* description =
    "Detected object set reader using CVAT XML format.\n\n"
    "CVAT XML format contains:\n"
    "  - <annotations> root element\n"
    "  - <image> elements with id, name, width, height\n"
    "  - <box> elements with label, xtl, ytl, xbr, ybr\n"
    "  - <polygon> elements with label and points\n\n"
    "Coordinates are in absolute pixels.";

  read_detected_object_set_cvat();
  virtual ~read_detected_object_set_cvat();

  virtual kwiver::vital::config_block_sptr get_configuration() const;
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual bool read_set( kwiver::vital::detected_object_set_sptr& set,
                         std::string& image_name );

private:
  virtual void new_stream();

  class priv;
  std::unique_ptr< priv > d;
};

} // end namespace

#endif // VIAME_CORE_READ_DETECTED_OBJECT_SET_CVAT_H
