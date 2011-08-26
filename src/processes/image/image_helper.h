/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_IMAGE_IMAGE_HELPER_H
#define VISTK_PROCESSES_IMAGE_IMAGE_HELPER_H

#include "image-config.h"

#include <vistk/pipeline/process.h>
#include <vistk/pipeline/types.h>

#include <boost/filesystem/path.hpp>
#include <boost/function.hpp>

#include <string>

namespace vistk
{

/// The type for the type for pixels in an image.
typedef std::string pixtype_t;

/// The type for file paths.
typedef boost::filesystem::path path_t;

/// The type of a function which reads an image from a file.
typedef boost::function<datum_t (path_t const&)> read_func_t;
/// The type of a function which writes in an image to a file.
typedef boost::function<void (path_t const&, datum_t const&)> write_func_t;

/// The type of a function which turns an image into grayscale.
typedef boost::function<datum_t (datum_t const&)> gray_func_t;
/// The type of a function which crops an image.
typedef boost::function<datum_t (datum_t const&, size_t, size_t, size_t, size_t)> crop_func_t;

/**
 * \class pixtypes "image_helper.h"
 *
 * \brief Names for common pixtypes.
 */
class VISTK_PROCESSES_IMAGE_NO_EXPORT pixtypes
{
  public:
    /**
     * \brief The pixtype for byte images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with bytes for pixels.
     */
    static pixtype_t const& pixtype_byte();
    /**
     * \brief The pixtype for floating images.
     *
     * \note This is a function to enforce static initialization orders.
     *
     * \returns The pixtype for images with floats for pixels.
     */
    static pixtype_t const& pixtype_float();
};

/**
 * \class image_helper "image_helper.h"
 *
 * \brief Helper class to help with managing image types.
 */
template <class PixType>
class VISTK_PROCESSES_IMAGE_NO_EXPORT image_helper
{
  public:
    /**
     * \struct port_types
     *
     * \brief Port types for images.
     */
    template <bool Grayscale = false, bool Alpha = false>
    struct port_types
    {
      /// The port type for the image.
      static process::port_type_t const type;
    };
};

/**
 * \brief Port types for images with different parameters.
 *
 * \param pixtype The type for pixels.
 * \param grayscale True if the images are grayscale, false otherwise.
 * \param alpha True if the images have an alpha channel, false otherwise.
 *
 * \returns The port type for images of the given information.
 */
process::port_type_t VISTK_PROCESSES_IMAGE_NO_EXPORT port_type_for_pixtype(pixtype_t const& pixtype, bool grayscale, bool alpha = false);

/**
 * \brief A reading function for images of the given type.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to reading \p pixtype images from a file.
 */
read_func_t VISTK_PROCESSES_IMAGE_NO_EXPORT read_for_pixtype(pixtype_t const& pixtype);
/**
 * \brief A writing function for images of the given type.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to writing \p pixtype images to a file.
 */
write_func_t VISTK_PROCESSES_IMAGE_NO_EXPORT write_for_pixtype(pixtype_t const& pixtype);

/**
 * \brief A grayscale conversion for images of the given type.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to converting \p pixtype images to grayscale.
 */
gray_func_t VISTK_PROCESSES_IMAGE_NO_EXPORT gray_for_pixtype(pixtype_t const& pixtype);
/**
 * \brief A cropping function for images of the given type.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to cropping \p pixtype images.
 */
crop_func_t VISTK_PROCESSES_IMAGE_NO_EXPORT crop_for_pixtype(pixtype_t const& pixtype);

}

#endif // VISTK_PROCESSES_IMAGE_IMAGE_HELPER_H
