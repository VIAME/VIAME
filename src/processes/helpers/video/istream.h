/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PROCESSES_HELPER_VIDEO_ISTREAM_H
#define VISTK_PROCESSES_HELPER_VIDEO_ISTREAM_H

#include <processes/helpers/image/pixtypes.h>

#include <vistk/pipeline/types.h>

#include <vistk/utilities/path.h>

#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include <vidl/vidl_istream.h>

#include <set>

/**
 * \file istream.h
 *
 * \brief Types and functions to help reading videos from a file.
 */

namespace vistk
{

/// An istream.
typedef boost::shared_ptr<vidl_istream> istream_t;

/// The type of a function which reads an image from a stream.
typedef boost::function<datum_t (istream_t const&)> istream_read_func_t;

/// The type for an implementation of an istream.
typedef std::string istream_impl_t;
/// The type for a collection of implementations.
typedef std::set<istream_impl_t> istream_impls_t;

/**
 * \brief The default implementation.
 *
 * \returns The default implementation.
 */
istream_impl_t const& default_istream_impl();

/**
 * \brief The known implementation types.
 *
 * \returns Implementations which are known.
 */
istream_impls_t known_istream_impls();

/**
 * \brief An istream for a given implementation.
 *
 * \param path The path to open.
 * \param grayscale Whether the output should be grayscale or not.
 * \param alpha Whether the output has an alpha channel or not.
 *
 * \returns An istream for the given implementation.
 */
istream_t istream_for_impl(istream_impl_t const& impl, path_t const& path);

/**
 * \brief An istream function for pixtypes of a given type.
 *
 * \param pixtype The type for pixels.
 *
 * \returns A function to read \p pixtype images from a video.
 */
istream_read_func_t istream_read_for_pixtype(pixtype_t const& pixtype, bool grayscale, bool alpha = false);

}

#endif // VISTK_PROCESSES_HELPER_VIDEO_ISTREAM_H
