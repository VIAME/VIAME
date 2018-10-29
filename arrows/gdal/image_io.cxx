/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \brief GDAL image_io implementation
 */

#include "image_io.h"

#include <vital/exceptions/algorithm.h>
#include <vital/exceptions/io.h>
#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>

#include <gdal_priv.h>
#include <ogr_spatialref.h>

#include <sstream>

namespace kwiver {
namespace arrows {
namespace gdal {

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

vital::polygon::point_t apply_geo_transform(double gt[], double x, double y)
{
  vital::polygon::point_t retVal;
  retVal[0] = gt[0] + gt[1]*x + gt[2]*y;
  retVal[1] = gt[3] + gt[4]*x + gt[5]*y;
  return retVal;
}

/// Constructor
image_io
::image_io()
{
}

/// Destructor
image_io
::~image_io()
{
}

/// Load image image from the file
/**
 * \param filename the path to the file the load
 * \returns an image container refering to the loaded image
 */
vital::image_container_sptr
image_io
::load_(const std::string& filename) const
{
  GDALAllRegister();

  std::shared_ptr<GDALDataset> gdalDataset;

  gdalDataset.reset( static_cast<GDALDataset*>(
    GDALOpen( filename.c_str(), GA_ReadOnly ) ) );

  vital::image img;

  if ( !gdalDataset )
  {
    VITAL_THROW( vital::invalid_file, filename, "GDAL could not load file.");
  }
  else
  {
    // Get the size of the image and the number of bands
    auto imgWidth = gdalDataset->GetRasterXSize();
    auto imgHeight = gdalDataset->GetRasterYSize();
    auto imgDepth = gdalDataset->GetRasterCount();

    // Create a new vital image based on the GDAL raster type. For now just
    // load bands directly into channels in vital::image.
    // TODO: deal or provide warning if bands have different types.
    auto bandType = gdalDataset->GetRasterBand(1)->GetRasterDataType();
    switch (bandType)
    {
      case (GDT_Byte):
      {
        img = vital::image_of<vital::byte>(imgWidth, imgHeight, imgDepth);
        break;
      }
      case (GDT_UInt16):
      {
        img = vital::image_of<uint16_t>(imgWidth, imgHeight, imgDepth);
        break;
      }
      case (GDT_Int16):
      {
        img = vital::image_of<int16_t>(imgWidth, imgHeight, imgDepth);
        break;
      }
      case (GDT_UInt32):
      {
        img = vital::image_of<uint32_t>(imgWidth, imgHeight, imgDepth);
        break;
      }
      case (GDT_Int32):
      {
        img = vital::image_of<int32_t>(imgWidth, imgHeight, imgDepth);
        break;
      }
      case (GDT_Float32):
      {
        img = vital::image_of<float>(imgWidth, imgHeight, imgDepth);
        break;
      }
      case (GDT_Float64):
      {
        img = vital::image_of<double>(imgWidth, imgHeight, imgDepth);
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

    // Loop over bands and copy data
    CPLErr err;
    for (int i = 1; i <= imgDepth; ++i)
    {
      GDALRasterBand* band = gdalDataset->GetRasterBand(i);
      bandType = band->GetRasterDataType();
      err = band->RasterIO(GF_Read, 0, 0, imgWidth, imgHeight,
        static_cast<void*>(reinterpret_cast<GByte*>(
          img.first_pixel()) + (i-1)*img.d_step()*img.pixel_traits().num_bytes),
        imgWidth, imgHeight, bandType, 0, 0);
    }

    // Get and translate metadata
    vital::metadata_sptr md = std::make_shared<vital::metadata>();

    // Get geotransform and calculate corner points
    double geo_transform[6];
    gdalDataset->GetGeoTransform(geo_transform);

    OGRSpatialReference osrs;
    osrs.importFromWkt( gdalDataset->GetProjectionRef() );

    // If coordinate system available - calculate corner points.
    if ( osrs.GetAuthorityCode("GEOGCS") )
    {
      vital::polygon points;
      points.push_back( apply_geo_transform(geo_transform, 0, 0) );
      points.push_back( apply_geo_transform(geo_transform, 0, imgHeight) );
      points.push_back( apply_geo_transform(geo_transform, imgWidth, 0) );
      points.push_back( apply_geo_transform(geo_transform, imgWidth, imgHeight) );

      md->add( NEW_METADATA_ITEM( vital::VITAL_META_CORNER_POINTS,
        vital::geo_polygon( points, atoi( osrs.GetAuthorityCode("GEOGCS") ) ) ) );
    }

    // Get RPC metadata
    char** rpc_metadata = gdalDataset->GetMetadata("RPC");
    if (CSLCount(rpc_metadata) > 0)
    {
      for (int i = 0; rpc_metadata[i] != NULL; ++i)
      {
        add_rpc_metadata( rpc_metadata[i] , md );
      }
    }

    return std::make_shared<vital::simple_image_container>(img, md);
  }
}


/// Save image image to a file
/**
 * \param filename the path to the file to save.
 * \param data The image container refering to the image to write.
 */
void
image_io
::save_(const std::string& filename,
       vital::image_container_sptr data) const
{
  VITAL_THROW( vital::algorithm_exception, this->type_name(), this->impl_name(),
               "Saving to file not supported." );
}

} // end namespace gdal
} // end namespace arrows
} // end namespace kwiver
