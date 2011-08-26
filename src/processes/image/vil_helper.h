/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_IMAGE_VIL_HELPER_H
#define VISTK_PROCESSES_IMAGE_VIL_HELPER_H

#include "image-config.h"

#include <vistk/pipeline/process.h>
#include <vistk/pipeline/types.h>

#include <boost/filesystem/path.hpp>
#include <boost/function.hpp>

#include <string>

namespace vistk
{

typedef std::string pixtype_t;

typedef boost::filesystem::path path_t;

typedef boost::function<datum_t (path_t const&)> read_func_t;
typedef boost::function<void (path_t const&, datum_t const&)> write_func_t;

typedef boost::function<datum_t (datum_t const&)> gray_func_t;
typedef boost::function<datum_t (datum_t const&, size_t, size_t, size_t, size_t)> crop_func_t;

class VISTK_PROCESSES_IMAGE_NO_EXPORT pixtypes
{
  public:
    static pixtype_t const& pixtype_byte();
    static pixtype_t const& pixtype_float();
};

/**
 * \class vil_helper "vil_helper.h"
 *
 * \brief Helper class to help with managing vil types.
 */
template <class PixType>
class VISTK_PROCESSES_IMAGE_NO_EXPORT vil_helper
{
  public:
    template <bool Grayscale = false, bool Alpha = false>
    struct port_types
    {
      static process::port_type_t const type;
    };
};

process::port_type_t VISTK_PROCESSES_IMAGE_NO_EXPORT port_type_for_pixtype(pixtype_t const& pixtype, bool grayscale, bool alpha = false);

read_func_t VISTK_PROCESSES_IMAGE_NO_EXPORT read_for_pixtype(pixtype_t const& pixtype);
write_func_t VISTK_PROCESSES_IMAGE_NO_EXPORT write_for_pixtype(pixtype_t const& pixtype);

gray_func_t VISTK_PROCESSES_IMAGE_NO_EXPORT gray_for_pixtype(pixtype_t const& pixtype);
crop_func_t VISTK_PROCESSES_IMAGE_NO_EXPORT crop_for_pixtype(pixtype_t const& pixtype);

}

#endif // VISTK_PROCESSES_IMAGE_VIL_HELPER_H
