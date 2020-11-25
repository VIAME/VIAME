// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief VTK depth estimation utility functions.
*/

#include <arrows/vtk/depth_utils.h>

#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>

namespace kwiver {
namespace arrows {
namespace vtk {

/// Convert an estimated depth image into vtkImageData
vtkSmartPointer<vtkImageData>
depth_to_vtk(kwiver::vital::image_of<double> const& depth_img,
             kwiver::vital::image_of<unsigned char> const& color_img,
             kwiver::vital::bounding_box<int> const& crop_box,
             kwiver::vital::image_of<double> const& uncertainty_img,
             kwiver::vital::image_of<unsigned char> const& mask_img)
{
  const bool valid_crop = crop_box.is_valid();
  const int i0 = valid_crop ? crop_box.min_x() : 0;
  const int j0 = valid_crop ? crop_box.min_y() : 0;
  const int ni = valid_crop ? crop_box.width() : depth_img.width();
  const int nj = valid_crop ? crop_box.height() : depth_img.height();

  if (depth_img.size() == 0 ||
      color_img.size() == 0 ||
      depth_img.width() + i0 > color_img.width() ||
      depth_img.height() + j0 > color_img.height())
  {
    return nullptr;
  }

  vtkNew<vtkDoubleArray> uncertainty;
  uncertainty->SetName("Uncertainty");
  uncertainty->SetNumberOfValues(ni * nj);

  vtkNew<vtkDoubleArray> weight;
  weight->SetName("Weight");
  weight->SetNumberOfValues(ni * nj);

  vtkNew<vtkUnsignedCharArray> color;
  color->SetName("Color");
  color->SetNumberOfComponents(3);
  color->SetNumberOfTuples(ni * nj);

  vtkNew<vtkDoubleArray> depths;
  depths->SetName("Depths");
  depths->SetNumberOfComponents(1);
  depths->SetNumberOfTuples(ni * nj);

  vtkNew<vtkIntArray> crop;
  crop->SetName("Crop");
  crop->SetNumberOfComponents(1);
  crop->SetNumberOfValues(4);
  crop->SetValue(0, i0);
  crop->SetValue(1, ni);
  crop->SetValue(2, j0);
  crop->SetValue(3, nj);

  vtkIdType pt_id = 0;

  const bool has_mask = mask_img.size() > 0;
  const bool has_uncertainty = uncertainty_img.size() > 0;
  for (int y = nj - 1; y >= 0; y--)
  {
    for (int x = 0; x < ni; x++)
    {
      depths->SetValue(pt_id, depth_img(x, y));

      color->SetTuple3(pt_id,
                       (int)color_img(x + i0, y + j0, 0),
                       (int)color_img(x + i0, y + j0, 1),
                       (int)color_img(x + i0, y + j0, 2));

      const double w = (has_mask && mask_img(x, y) > 127) ? 1.0 : 0.0;
      weight->SetValue(pt_id, w);

      uncertainty->SetValue(pt_id,
        has_uncertainty ? uncertainty_img(x, y) : 0.0);

      pt_id++;
    }
  }

  auto imageData = vtkSmartPointer<vtkImageData>::New();
  imageData->SetSpacing(1, 1, 1);
  imageData->SetOrigin(0, 0, 0);
  imageData->SetDimensions(ni, nj, 1);
  imageData->GetPointData()->AddArray(depths.Get());
  imageData->GetPointData()->AddArray(color.Get());
  imageData->GetPointData()->AddArray(uncertainty.Get());
  imageData->GetPointData()->AddArray(weight.Get());
  imageData->GetFieldData()->AddArray(crop.Get());
  return imageData;
}

} //end namespace vtk
} //end namespace arrows
} //end namespace kwiver
