/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "image_reader_process.h"

#include <processes/helpers/image/format.h>
#include <processes/helpers/image/read.h>

#include <vistk/utilities/path.h>
#include <vistk/utilities/timestamp.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>

#include <string>

/**
 * \file image_reader_process.cxx
 *
 * \brief Implementation of the image reader process.
 */

namespace vistk
{

class image_reader_process::priv
{
  public:
    priv(read_func_t func);
    ~priv();

    read_func_t const read;

    timestamp::frame_t frame;

    static config::key_t const config_pixtype;
    static config::key_t const config_pixfmt;
    static config::value_t const default_pixtype;
    static config::value_t const default_pixfmt;
    static port_t const port_input;
    static port_t const port_output;
    static port_t const port_output_ts;
};

config::key_t const image_reader_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const image_reader_process::priv::config_pixfmt = config::key_t("pixfmt");
config::value_t const image_reader_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
config::value_t const image_reader_process::priv::default_pixfmt = config::value_t(pixfmts::pixfmt_rgb());
process::port_t const image_reader_process::priv::port_input = port_t("path");
process::port_t const image_reader_process::priv::port_output = port_t("image");
process::port_t const image_reader_process::priv::port_output_ts = port_t("timestamp");

image_reader_process
::image_reader_process(config_t const& config)
  : process(config)
  , d()
{
  declare_configuration_key(
    priv::config_pixtype,
    priv::default_pixtype,
    config::description_t("The pixel type of the input images."));
  declare_configuration_key(
    priv::config_pixfmt,
    priv::default_pixfmt,
    config::description_t("The pixel format of the input images."));

  pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
  pixfmt_t const pixfmt = config_value<pixfmt_t>(priv::config_pixfmt);

  port_type_t const port_type_output = port_type_for_pixtype(pixtype, pixfmt);

  port_flags_t const none;
  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_input,
    port_type_t("path"),
    required,
    port_description_t("The path to the image to read."));

  declare_output_port(
    priv::port_output,
    port_type_output,
    required,
    port_description_t("The images that are read in."));
  declare_output_port(
    priv::port_output_ts,
    "timestamp",
    none,
    port_description_t("The timestamp for the image."));
}

image_reader_process
::~image_reader_process()
{
}

void
image_reader_process
::_configure()
{
  // Configure the process.
  {
    pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);

    read_func_t const func = read_for_pixtype(pixtype);

    d.reset(new priv(func));
  }

  if (!d->read)
  {
    static std::string const reason = "A read function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }

  process::_configure();
}

void
image_reader_process
::_step()
{
  vistk::path_t const path = grab_from_port_as<vistk::path_t>(priv::port_input);

  datum_t const dat = d->read(path);

  ++d->frame;

  timestamp const ts = timestamp(d->frame);

  datum_t const dat_ts = datum::new_datum(ts);

  push_datum_to_port(priv::port_output, dat);
  push_datum_to_port(priv::port_output_ts, dat_ts);

  process::_step();
}

image_reader_process::priv
::priv(read_func_t func)
  : read(func)
  , frame(0)
{
}

image_reader_process::priv
::~priv()
{
}

}
