// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Header for VTK depth estimation utility functions.
*/

#ifndef KWIVER_ARROWS_VTK_DEPTH_UTILS_H_
#define KWIVER_ARROWS_VTK_DEPTH_UTILS_H_

#include <arrows/vtk/kwiver_algo_vtk_export.h>

#include <vital/algo/image_io.h>
#include <vital/io/camera_io.h>
#include <vital/types/bounding_box.h>
#include <vital/types/image.h>

#include <vtkImageData.h>
#include <vtkSmartPointer.h>

namespace kwiver {
namespace arrows {
namespace vtk {

/// Convert an estimated depth image into vtkImageData
/**
 * The vtkImageData contains not just the depth image, but also the
 * associated uncertainty, color, and weight images as well as a bounding
 * box specifying how the image was cropped from the source image.
 * The depth image may be cropped relative to the source color image.
 * If so, the crop_box used for this crop should be specified.  The color
 * image should be the original dimensions and crop_box is used to crop out
 * corresponding color values to store in the output.  The optional uncertainty
 * and mask images, if provided, should have the same dimensions as the depth
 * image.
 *
 * \param depth_img       The floating point depth value at each pixel
 * \param color_img       The source uncropped RGB image corresonding to depth_img
 * \param uncertainty_img The optional standard deviation of the depth at each pixel
 * \param mask_img        The mask of which pixels to ignore
 * \param crop_box        The bounding box used to crop depth_img from color_img
 */
KWIVER_ALGO_VTK_EXPORT
vtkSmartPointer<vtkImageData>
depth_to_vtk(kwiver::vital::image_of<double> const& depth_img,
             kwiver::vital::image_of<unsigned char> const& color_img,
             kwiver::vital::bounding_box<int> const& crop_box = {},
             kwiver::vital::image_of<double> const& uncertainty_img = {},
             kwiver::vital::image_of<unsigned char> const& mask_img = {});

/// Load the depth map saved as a VTK array
/**
 * Read in the information stored in a single output file
 *
 * \param [in] filename         The file to load from
 * \param [out] crop            The bounds that the data should be cropped to
 * \param [out] depth_out       Depth values
 * \param [out] weight_out      Weights for each pixel (mask)
 * \param [out] uncertainty_out Uncertainty for each pixel
 * \param [out] color_out       Color for each pixel
 */
KWIVER_ALGO_VTK_EXPORT
void
load_depth_map(const std::string& filename,
               vital::bounding_box<int>& crop,
               kwiver::vital::image_container_sptr& depth_out,
               kwiver::vital::image_container_sptr& weight_out,
               kwiver::vital::image_container_sptr& uncertainty_out,
               kwiver::vital::image_container_sptr& color_out);

/// Convert a volume and metadata to the VTK format
/**
 * \param [in] volume   The volumetric data to convert
 * \param [in] origin   The local coordinate system origin of the data
 * \param [in] spacing  The voxel spacing per axis
 */
KWIVER_ALGO_VTK_EXPORT
vtkSmartPointer<vtkImageData>
volume_to_vtk(kwiver::vital::image_container_sptr volume,
              kwiver::vital::vector_3d const& origin,
              kwiver::vital::vector_3d const& spacing);

} //end namespace vtk
} //end namespace arrows
} //end namespace kwiver

#endif
