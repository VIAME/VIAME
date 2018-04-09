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

#include <gdal_priv.h>

namespace kwiver {
namespace arrows {
namespace gdal {

/// Private implementation class
class image_io::priv
{
public:
  /// Constructor
  priv() : m_logger( vital::get_logger( "arrows.gdal.image_io" ) ) {}

  vital::logger_handle_t m_logger;
};

/// Constructor
image_io
::image_io()
: d_(new priv)
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
    LOG_ERROR( d_->m_logger, "Unable to load data from " << filename );
    throw vital::image_exception();
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
        LOG_ERROR( d_->m_logger, "Unknown or unsupported pixal type: "
                  << bandType );
        throw vital::image_type_mismatch_exception("kwiver::arrows::gdal::image_io::load()");
        break;
      }
    }
  }

  // Get and translate metadata
  char** mdDomains = gdalDataset->GetMetadataDomainList();
  if (CSLCount(mdDomains) > 0)
  {
    for (int i = 0; mdDomains[i] != NULL; ++i)
    {
      std::cout << mdDomains[i] << std::endl;
    }
  }

  return std::make_shared<vital::simple_image_container>(img);
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

}

} // end namespace gdal
} // end namespace arrows
} // end namespace kwiver
