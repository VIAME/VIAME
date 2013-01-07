/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "read_video_gravl_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/read.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <gravl/core/api/data_block.h>
#include <gravl/core/api/frame_ptr.h>
#include <gravl/core/api/image.h>
#include <gravl/core/api/image_data.h>
#include <gravl/core/api/resource_ptr.h>

#include <gravl/raf/raf.h>

#include <vil/vil_image_view.h>

#include <string>

/**
 * \file read_video_gravl_process.cxx
 *
 * \brief Implementation of the read video gravl process.
 */

static size_t compute_block_size(gravl::image::dimension dim,
                                 gravl::image::stride s);
template <typename T> static T* compute_block_start(
  T* top_left, gravl::image::dimension dim, gravl::image::stride s);
template <typename T> static T* compute_top_left(
  T* top_left, gravl::image::dimension dim, gravl::image::stride s);

namespace vistk
{

class read_video_gravl_process::priv
{
  public:
    priv(std::string const& input_uri);
    ~priv();

    std::string const uri;

    gravl::resource_ptr resource;
    gravl::data_block* video;

    static config::key_t const config_pixtype;
    static config::key_t const config_pixfmt;
    static config::key_t const config_uri;
    static config::key_t const config_verify;
    static config::value_t const default_pixtype;
    static config::value_t const default_pixfmt;
    static config::value_t const default_verify;
    static port_t const port_output;
};

config::key_t const read_video_gravl_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const read_video_gravl_process::priv::config_pixfmt = config::key_t("pixfmt");
config::key_t const read_video_gravl_process::priv::config_uri = config::key_t("input");
config::value_t const read_video_gravl_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
config::value_t const read_video_gravl_process::priv::default_pixfmt = config::value_t(pixfmts::pixfmt_rgb());
process::port_t const read_video_gravl_process::priv::port_output = port_t("image");

read_video_gravl_process
::read_video_gravl_process(config_t const& config)
  : process(config)
{
  declare_configuration_key(
    priv::config_pixtype,
    priv::default_pixtype,
    config::description_t("The pixel type of the input images."));
  declare_configuration_key(
    priv::config_pixfmt,
    priv::default_pixfmt,
    config::description_t("The pixel format of the input images."));
  declare_configuration_key(
    priv::config_uri,
    config::value_t(),
    config::description_t("The URI of the resource."));

  pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
  pixfmt_t const pixfmt = config_value<pixfmt_t>(priv::config_pixfmt);

  port_type_t const port_type_output = port_type_for_pixtype(pixtype, pixfmt);

  port_flags_t required;

  required.insert(flag_required);

  declare_output_port(
    priv::port_output,
    port_type_output,
    required,
    port_description_t("The images that are read in."));
}

read_video_gravl_process
::~read_video_gravl_process()
{
}

void
read_video_gravl_process
::_configure()
{
  // Configure the process.
  {
    pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
    std::string const uri = config_value<std::string>(priv::config_uri);

    d.reset(new priv(uri));
  }

  if (d->uri.empty())
  {
    static std::string const reason = "The URI given was empty";
    config::value_t const value = config::value_t(d->uri);

    throw invalid_configuration_value_exception(name(), priv::config_uri, value, reason);
  }

  d->resource = gravl::raf::get_resource(d->uri.c_str());
  if (!d->resource)
  {
    std::string const reason = "Failed to open the resource: " + d->uri;

    throw invalid_configuration_exception(name(), reason);
  }

  d->video = d->resource.get_data<gravl::data_block>();
  if (!d->video)
  {
    static std::string const reason = "Failed to obtain data_block from resource";

    throw invalid_configuration_exception(name(), reason);
  }

  process::_configure();
}

void
read_video_gravl_process
::_init()
{
  d->video->rewind();

  process::_init();
}

void
read_video_gravl_process
::_step()
{
  datum_t dat;

  if (d->video->at_end())
  {
    mark_process_as_complete();
    dat = datum::complete_datum();
  }
  else if (!d->video)
  {
    static datum::error_t const err_string = datum::error_t("Error with input file stream");

    dat = datum::error_datum(err_string);
  }
  else
  {
    gravl::frame_ptr const frame = d->video->current_frame();
    gravl::image_data const* const image_data = frame.get_const_data<gravl::image_data>();
    gravl::image const image = (image_data ? image_data->pixels() : gravl::image());

    if (!image || image.format() != gravl::image::format_of<uint8_t>())
    {
      dat = datum::empty_datum();
    }
    else
    {
      gravl::image::dimension const dim = image.dimensions();
      gravl::image::stride const ps = image.strides();
      uint8_t const* const top_left = image.data<uint8_t>();
      size_t const size = compute_block_size(dim, ps);

      // Ugh, there is no way to create a vil_image_view from existing data
      // without arranging for said existing data to stick around, so stuck
      // having to copy the data (again) :-(
      vil_memory_chunk* const mem = new vil_memory_chunk(size, VIL_PIXEL_FORMAT_BYTE);
      uint8_t* const buffer = reinterpret_cast<uint8_t*>(mem->data());
      memcpy(buffer, compute_block_start(top_left, dim, ps), size);

      vil_image_view<vxl_byte> const vil(vil_memory_chunk_sptr(mem),
                                         compute_top_left(buffer, dim, ps),
                                         dim.width, dim.height, dim.planes,
                                         ps.width, ps.height, ps.planes);
      dat = datum::new_datum(vil);
    }

    d->video->advance();
  }

  push_datum_to_port(priv::port_output, dat);

  process::_step();
}

read_video_gravl_process::priv
::priv(std::string const& input_uri)
  : uri(input_uri)
{
}

read_video_gravl_process::priv
::~priv()
{
}

}

size_t
sabs(ptrdiff_t a)
{
  return static_cast<size_t>((a < 0 ? -a : a));
}

size_t
compute_block_size(gravl::image::dimension dim, gravl::image::stride s)
{
  return ((dim.width  - 1) * sabs(s.width)) +
         ((dim.height - 1) * sabs(s.height)) +
         ((dim.planes - 1) * sabs(s.planes)) + 1;
}

ptrdiff_t
compute_offset(gravl::image::dimension dim, gravl::image::stride s)
{
  ptrdiff_t result = 0;
  if (s.width < 0)
  {
    result -= s.width * (dim.width - 1);
  }
  if (s.height < 0)
  {
    result -= s.height * (dim.height - 1);
  }
  if (s.planes < 0)
  {
    result -= s.planes * (dim.planes - 1);
  }
  return result;
}

template <typename T>
T*
compute_block_start(
  T* top_left, gravl::image::dimension dim, gravl::image::stride s)
{
  return top_left - compute_offset(dim, s);
}

template <typename T>
T*
compute_top_left(
  T* top_left, gravl::image::dimension dim, gravl::image::stride s)
{
  return top_left + compute_offset(dim, s);
}
