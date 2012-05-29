/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_IMAGE_WARP_IMAGE_H
#define VISTK_IMAGE_WARP_IMAGE_H

#include "image-config.h"

#include <vistk/utilities/homography.h>

#include <boost/optional.hpp>
#include <boost/scoped_ptr.hpp>

#include <vil/vil_image_view.h>

/**
 * \file warp_image.h
 *
 * \brief Declaration of the function to warp images.
 */

namespace vistk
{

/**
 * \class warp_image "warp_image.h" <vistk/image/warp_image.h>
 *
 * \brief Warps an image given a transformation.
 */
template <typename PixType>
class VISTK_IMAGE_EXPORT warp_image
{
  public:
    /// The type of an image for warping.
    typedef vil_image_view<PixType> image_t;
    /// The type of a transformation matrix.
    typedef homography_base::transform_t transform_t;
    /// The type of a mask for where the destination pixels were set.
    typedef vil_image_view<bool> mask_t;
    /// The type of a pixel.
    typedef PixType pixel_t;
    /// The type of a fill value.
    typedef boost::optional<pixel_t> fill_t;

    /**
     * \brief Constructor.
     *
     * \param dest_width The width of the destination image.
     * \param dest_height The height of the destination image.
     * \param dest_planes The number of planes in the destination image.
     * \param fill_value The value to fill pixels not mapped with.
     */
    warp_image(size_t dest_width, size_t dest_height, size_t dest_planes, fill_t const& fill_value = fill_t());
    /**
     * \brief Destructor.
     */
    ~warp_image();

    /**
     * \brief Resets the mask.
     */
    void clear_mask();

    /**
     * \brief The mask from the last warp.
     *
     * \returns The mask that resulted from the last warp.
     */
    mask_t mask() const;

    /**
     * \brief Warps an image to the destination image.
     *
     * \param image The image to warp.
     * \param transform The transform to use to warp.
     *
     * \returns The destination image with the image warped to it.
     */
    image_t operator () (image_t const& image, transform_t const& transform) const;
  private:
    class priv;
    boost::scoped_ptr<priv> d;
};

}

#endif // VISTK_IMAGE_WARP_IMAGE_H
