/*ckwg +29
 * Copyright 2018-2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief GDAL image_container implementation
 */

#include "image_container.h"

#include <vital/exceptions/io.h>
#include <vital/types/metadata_traits.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace gdal {

// ----------------------------------------------------------------------------
void add_rpc_metadata(char* raw_md, vital::metadata_sptr md)
{
  std::istringstream md_string(raw_md);

  // Get the key
  std::string key;
  if ( !std::getline( md_string, key, '=') )
  {
    return;
  }

  // Get the value
  std::string value;
  if ( !std::getline( md_string, value, '=') )
  {
    return;
  }

#define MAP_METADATA_SCALAR( GN, KN )                       \
if ( key == #GN)                                            \
{                                                           \
  md->add( NEW_METADATA_ITEM(                               \
    vital::VITAL_META_RPC_ ## KN, std::stod( value ) ) ) ;  \
}                                                           \

#define MAP_METADATA_COEFF( GN, KN )          \
if ( key == #GN)                              \
{                                             \
  md->add( NEW_METADATA_ITEM(                 \
    vital::VITAL_META_RPC_ ## KN, value ) );  \
}                                             \

  MAP_METADATA_SCALAR( HEIGHT_OFF,   HEIGHT_OFFSET )
  MAP_METADATA_SCALAR( HEIGHT_SCALE, HEIGHT_SCALE )
  MAP_METADATA_SCALAR( LONG_OFF,     LONG_OFFSET )
  MAP_METADATA_SCALAR( LONG_SCALE,   LONG_SCALE )
  MAP_METADATA_SCALAR( LAT_OFF,      LAT_OFFSET )
  MAP_METADATA_SCALAR( LAT_SCALE,    LAT_SCALE )
  MAP_METADATA_SCALAR( LINE_OFF,     ROW_OFFSET )
  MAP_METADATA_SCALAR( LINE_SCALE,   ROW_SCALE )
  MAP_METADATA_SCALAR( SAMP_OFF,     COL_OFFSET )
  MAP_METADATA_SCALAR( SAMP_SCALE,   COL_SCALE )

  MAP_METADATA_COEFF( LINE_NUM_COEFF, ROW_NUM_COEFF )
  MAP_METADATA_COEFF( LINE_DEN_COEFF, ROW_DEN_COEFF )
  MAP_METADATA_COEFF( SAMP_NUM_COEFF, COL_NUM_COEFF )
  MAP_METADATA_COEFF( SAMP_DEN_COEFF, COL_DEN_COEFF )

#undef MAP_METADATA_SCALAR
#undef MAP_METADATA_COEFF
}

void add_nitf_metadata(char* raw_md, vital::metadata_sptr md)
{
  std::istringstream md_string(raw_md);

  // Get the key
  std::string key;
  if ( !std::getline( md_string, key, '=') )
  {
    return;
  }

  // Get the value
  std::string value;
  if ( !std::getline( md_string, value, '=') )
  {
    return;
  }

#define MAP_METADATA_COEFF( GN, KN )          \
if ( key == #GN)                              \
{                                             \
  md->add( NEW_METADATA_ITEM(                 \
    vital::VITAL_META_NITF_ ## KN, value ) );  \
}                                             \

  MAP_METADATA_COEFF( NITF_IDATIM, IDATIM )
  MAP_METADATA_COEFF( NITF_BLOCKA_FRFC_LOC_01, BLOCKA_FRFC_LOC_01 )
  MAP_METADATA_COEFF( NITF_BLOCKA_FRLC_LOC_01, BLOCKA_FRLC_LOC_01 )
  MAP_METADATA_COEFF( NITF_BLOCKA_LRLC_LOC_01, BLOCKA_LRLC_LOC_01 )
  MAP_METADATA_COEFF( NITF_BLOCKA_LRFC_LOC_01, BLOCKA_LRFC_LOC_01 )
  MAP_METADATA_COEFF( NITF_IMAGE_COMMENTS, IMAGE_COMMENTS )

#undef MAP_METADATA_SCALAR
#undef MAP_METADATA_COEFF
}

vital::polygon::point_t apply_geo_transform(double gt[], double x, double y)
{
  vital::polygon::point_t retVal;
  retVal[0] = gt[0] + gt[1]*x + gt[2]*y;
  retVal[1] = gt[3] + gt[4]*x + gt[5]*y;
  return retVal;
}

// ----------------------------------------------------------------------------
image_container
::image_container(const std::string& filename)
{
  GDALAllRegister();

  gdal_dataset_.reset(
    static_cast<GDALDataset*>(GDALOpen( filename.c_str(), GA_ReadOnly ) ) );

  if ( !gdal_dataset_ )
  {
    VITAL_THROW( vital::invalid_file, filename, "GDAL could not load file.");
  }

  // Get image pixel traits based on the GDAL raster type.
  // TODO: deal or provide warning if bands have different types.
  auto bandType = gdal_dataset_->GetRasterBand(1)->GetRasterDataType();
  switch (bandType)
  {
    case (GDT_Byte):
    {
      pixel_traits_ = vital::image_pixel_traits_of<uint8_t>();
      break;
    }
    case (GDT_UInt16):
    {
      pixel_traits_ = vital::image_pixel_traits_of<uint16_t>();
      break;
    }
    case (GDT_Int16):
    {
      pixel_traits_ = vital::image_pixel_traits_of<int16_t>();
      break;
    }
    case (GDT_UInt32):
    {
      pixel_traits_ = vital::image_pixel_traits_of<uint32_t>();
      break;
    }
    case (GDT_Int32):
    {
      pixel_traits_ = vital::image_pixel_traits_of<int32_t>();
      break;
    }
    case (GDT_Float32):
    {
      pixel_traits_ = vital::image_pixel_traits_of<float>();
      break;
    }
    case (GDT_Float64):
    {
      pixel_traits_ = vital::image_pixel_traits_of<double>();
      break;
    }
    default:
    {
      std::stringstream ss;
      ss << "kwiver::arrows::gdal::image_io::load(): "
         << "Unknown or unsupported pixal type: "
         << GDALGetDataTypeName(bandType);
      VITAL_THROW( vital::image_type_mismatch_exception, ss.str() );
      break;
    }
  }

  vital::metadata_sptr md = std::make_shared<vital::metadata>();

  md->add( NEW_METADATA_ITEM(
    kwiver::vital::VITAL_META_IMAGE_URI, filename ) );

  // Get geotransform and calculate corner points
  double geo_transform[6];
  gdal_dataset_->GetGeoTransform(geo_transform);

  OGRSpatialReference osrs;
  osrs.importFromWkt( gdal_dataset_->GetProjectionRef() );

  // If coordinate system available - calculate corner points.
  if ( osrs.GetAuthorityCode("GEOGCS") )
  {
    vital::polygon points;
    const double h = static_cast<double>(this->height());
    const double w = static_cast<double>(this->width());
    points.push_back( apply_geo_transform(geo_transform, 0, 0) );
    points.push_back( apply_geo_transform(geo_transform, 0, h ) );
    points.push_back( apply_geo_transform(geo_transform, w, 0) );
    points.push_back( apply_geo_transform(geo_transform, w, h ) );

    md->add( NEW_METADATA_ITEM( vital::VITAL_META_CORNER_POINTS,
      vital::geo_polygon( points, atoi( osrs.GetAuthorityCode("GEOGCS") ) ) ) );
  }

  // Get RPC metadata
  char** rpc_metadata = gdal_dataset_->GetMetadata("RPC");
  if (CSLCount(rpc_metadata) > 0)
  {
    for (int i = 0; rpc_metadata[i] != NULL; ++i)
    {
      add_rpc_metadata( rpc_metadata[i] , md );
    }
  }

  // Get NITF metadata
  char** nitf_metadata = gdal_dataset_->GetMetadata("");
  if (CSLCount(nitf_metadata) > 0)
  {
    for (int i = 0; nitf_metadata[i] != NULL; ++i)
    {
      add_nitf_metadata( nitf_metadata[i] , md );
    }
  }

  this->set_metadata( md );
}

char**
image_container::get_raw_metadata_for_domain(const char *domain)
{
  return this->gdal_dataset_->GetMetadata(domain);
}

// ----------------------------------------------------------------------------
/// The size of the image data in bytes
size_t
image_container
::size() const
{
  return width() * height() * depth() * pixel_traits_.num_bytes;
}

// ----------------------------------------------------------------------------
/// Get image. Unlike other image containers must allocate memory
vital::image
image_container
::get_image() const
{
  vital::image img( width(), height(), depth(), false, pixel_traits_ );

  // Loop over bands and copy data
  CPLErr err;
  for (size_t i = 1; i <= depth(); ++i)
  {
    GDALRasterBand* band = gdal_dataset_->GetRasterBand(static_cast<int>(i));
    auto bandType = band->GetRasterDataType();
    const int h = static_cast<int>(this->height());
    const int w = static_cast<int>(this->width());
    err = band->RasterIO(GF_Read, 0, 0, w, h,
      static_cast<void*>(reinterpret_cast<GByte*>(
        img.first_pixel()) + (i-1)*img.d_step()*img.pixel_traits().num_bytes),
      w, h, bandType, 0, 0);
    // TODO Error checking on return value
    // this line silences unused variable warnings
    (void) err;
  }

  return img;
}

// ----------------------------------------------------------------------------
/// Get cropped view of image. Unlike other image containers must allocate memory
vital::image
image_container
::get_image(unsigned x_offset, unsigned y_offset,
            unsigned width, unsigned height) const
{
  vital::image img( width, height, depth(), false, pixel_traits_ );

  // Loop over bands and copy data
  CPLErr err;
  for (size_t i = 1; i <= depth(); ++i)
  {
    GDALRasterBand* band = gdal_dataset_->GetRasterBand(i);
    auto bandType = band->GetRasterDataType();
    err = band->RasterIO(GF_Read, x_offset, y_offset, width, height,
      static_cast<void*>(reinterpret_cast<GByte*>(
        img.first_pixel()) + (i-1)*img.d_step()*img.pixel_traits().num_bytes),
      width, height, bandType, 0, 0);

    (void) err;
  }

  return img;
}

} // end namespace gdal
} // end namespace arrows
} // end namespace kwiver
