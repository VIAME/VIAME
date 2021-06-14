// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief VTK depth estimation utility functions.
*/

#include <arrows/vtk/depth_utils.h>

#include <vital/types/bounding_box.h>

#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkXMLImageDataReader.h>

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

      const double w = (!has_mask || mask_img(x, y) > 127) ? 1.0 : 0.0;
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

//-----------------------------------------------------------------------------
template <typename T>
kwiver::vital::image_container_sptr
to_kwiver(vtkDataArray const* data, int const dims[3])
{
  using vtkDataArrayT = vtkAOSDataArrayTemplate<T>;
  vtkDataArrayT const* vtk_array = dynamic_cast<vtkDataArrayT const*>(data);
  if (!vtk_array)
  {
    return nullptr;
  }

  kwiver::vital::image_container_sptr output_img_ptr;

  if (vtk_array != nullptr)
  {
    kwiver::vital::image_of<T> output_img(dims[0], dims[1], dims[2]);
    vtkIdType pt_id = 0;
    // VTK images are stored with (0,0) at the bottom left instead of top left
    // so flip vertically when copying memory
    for (int y = dims[1]-1; y >= 0; y--)
    {
      for (int x = 0; x < dims[0]; x++)
      {
        output_img(x, y) = vtk_array->GetValue(pt_id);
        pt_id++;
      }
    }
    output_img_ptr =
      std::make_shared<kwiver::vital::simple_image_container>(output_img);
  }

  return output_img_ptr;
}

//-----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
extract_image(vtkImageData* img,
              std::string const& array_name)
{
  vtkDataArray *data = img->GetPointData()->GetArray(array_name.c_str());
  if (!data)
  {
    return nullptr;
  }

  int dims[3];
  img->GetDimensions(dims);

  switch (data->GetDataType())
  {
  case VTK_UNSIGNED_CHAR:
    return to_kwiver<uint8_t>(data, dims);
  case VTK_SIGNED_CHAR:
    return to_kwiver<int8_t>(data, dims);
  case VTK_UNSIGNED_SHORT:
    return to_kwiver<uint16_t>(data, dims);
  case VTK_SHORT:
    return to_kwiver<int16_t>(data, dims);
  case VTK_UNSIGNED_INT:
    return to_kwiver<uint32_t>(data, dims);
  case VTK_INT:
    return to_kwiver<int32_t>(data, dims);
  case VTK_UNSIGNED_LONG:
    return to_kwiver<uint64_t>(data, dims);
  case VTK_LONG:
    return to_kwiver<int64_t>(data, dims);
  case VTK_FLOAT:
    return to_kwiver<float>(data, dims);
  case VTK_DOUBLE:
    return to_kwiver<double>(data, dims);
  default:
    break;
  }
  return nullptr;
}

//-----------------------------------------------------------------------------
void
load_depth_map(const std::string& filename,
               kwiver::vital::bounding_box<int>& crop,
               kwiver::vital::image_container_sptr& depth_out,
               kwiver::vital::image_container_sptr& weight_out,
               kwiver::vital::image_container_sptr& uncertainty_out,
               kwiver::vital::image_container_sptr& color_out)
{
  vtkNew<vtkXMLImageDataReader> depthReader;
  depthReader->SetFileName(filename.c_str());
  depthReader->Update();
  vtkImageData *img = depthReader->GetOutput();

  vtkIntArray* read_crop = static_cast<vtkIntArray*>(
    img->GetFieldData()->GetArray("Crop"));
  int i0, ni, j0, nj;
  if (read_crop)
  {
    i0 = read_crop->GetValue(0);
    ni = read_crop->GetValue(1);
    j0 = read_crop->GetValue(2);
    nj = read_crop->GetValue(3);
    // Construct the crop as {xmin, ymin, xmax, ymax}
  }
  else
  {
    i0 = j0 = 0;
    ni = img->GetDimensions()[0];
    nj = img->GetDimensions()[1];
  }
  crop = kwiver::vital::bounding_box<int>(j0, i0, j0+nj, i0+ni);

  depth_out = extract_image(img, "Depths");
  weight_out = extract_image(img, "Weight");
  uncertainty_out = extract_image(img, "Uncertainty");
  color_out = extract_image(img, "Color");
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkImageData>
volume_to_vtk(kwiver::vital::image_container_sptr volume,
              kwiver::vital::vector_3d const& origin,
              kwiver::vital::vector_3d const& spacing)
{
  vtkSmartPointer<vtkImageData> grid = vtkSmartPointer<vtkImageData>::New();
  grid->SetOrigin(origin[0], origin[1], origin[2]);
  grid->SetDimensions(static_cast<int>(volume->width()),
                      static_cast<int>(volume->height()),
                      static_cast<int>(volume->depth()));
  grid->SetSpacing(spacing[0], spacing[1], spacing[2]);

  // initialize output
  vtkNew<vtkDoubleArray> vals;
  vals->SetName("reconstruction_scalar");
  vals->SetNumberOfComponents(1);
  vals->SetNumberOfTuples(volume->width() * volume->height() * volume->depth());

  vtkIdType pt_id = 0;
  const kwiver::vital::image &vol = volume->get_image();

  for (unsigned int k = 0; k < volume->depth(); k++)
  {
    for (unsigned int j = 0; j < volume->height(); j++)
    {
      for (unsigned int i = 0; i < volume->width(); i++)
      {
        vals->SetTuple1(pt_id++, vol.at<double>(i, j, k));
      }
    }
  }

  grid->GetPointData()->SetScalars(vals);
  grid->GetPointData()->GetAbstractArray(0)->SetName("reconstruction_scalar");
  return grid;
}

} //end namespace vtk
} //end namespace arrows
} //end namespace kwiver
