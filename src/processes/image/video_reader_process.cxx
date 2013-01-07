/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "video_reader_process.h"

#include <processes/helpers/video/istream.h>

#include <processes/helpers/image/format.h>

#include <vistk/utilities/path.h>

#include <vistk/pipeline/config.h>
#include <vistk/pipeline/datum.h>
#include <vistk/pipeline/process_exception.h>
#include <vistk/pipeline/stamp.h>

#include <boost/algorithm/string/join.hpp>

#include <vidl/vidl_istream.h>

#include <fstream>
#include <string>

/**
 * \file video_reader_process.cxx
 *
 * \brief Implementation of the video reader process.
 */

namespace vistk
{

class video_reader_process::priv
{
  public:
    priv(path_t const& input_path, istream_t const& istr, istream_read_func_t func, bool ver);
    ~priv();

    path_t const path;
    istream_t const istream;
    istream_read_func_t const read;
    bool const verify;

    static config::key_t const config_pixtype;
    static config::key_t const config_pixfmt;
    static config::key_t const config_path;
    static config::key_t const config_verify;
    static config::key_t const config_impl;
    static config::value_t const default_pixtype;
    static config::value_t const default_pixfmt;
    static config::value_t const default_verify;
    static config::value_t const default_impl;
    static port_t const port_output;
};

config::key_t const video_reader_process::priv::config_pixtype = config::key_t("pixtype");
config::key_t const video_reader_process::priv::config_pixfmt = config::key_t("pixfmt");
config::key_t const video_reader_process::priv::config_path = config::key_t("input");
config::key_t const video_reader_process::priv::config_verify = config::key_t("verify");
config::key_t const video_reader_process::priv::config_impl = config::key_t("impl");
config::value_t const video_reader_process::priv::default_pixtype = config::value_t(pixtypes::pixtype_byte());
config::value_t const video_reader_process::priv::default_pixfmt = config::value_t(pixfmts::pixfmt_rgb());
config::value_t const video_reader_process::priv::default_verify = config::value_t("false");
config::value_t const video_reader_process::priv::default_impl = config::value_t(default_istream_impl());
process::port_t const video_reader_process::priv::port_output = port_t("image");

static std::string const impl_sep = ", ";

video_reader_process
::video_reader_process(config_t const& config)
  : process(config)
  , d()
{
  istream_impls_t const impls = known_istream_impls();
  std::string impls_str = boost::join(impls, impl_sep);

  declare_configuration_key(
    priv::config_pixtype,
    priv::default_pixtype,
    config::description_t("The pixel type of the input images."));
  declare_configuration_key(
    priv::config_pixfmt,
    priv::default_pixfmt,
    config::description_t("The pixel format of the input images."));
  declare_configuration_key(
    priv::config_path,
    config::value_t(),
    config::description_t("The input file with a list of images to read."));
  declare_configuration_key(
    priv::config_verify,
    priv::default_verify,
    config::description_t("If \'true\', the paths in the input file will checked that they can be read."));
  declare_configuration_key(
    priv::config_impl,
    priv::default_impl,
    config::description_t("The implementation to use for reading the input. Known: " + impls_str));

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

video_reader_process
::~video_reader_process()
{
}

void
video_reader_process
::_configure()
{
  // Configure the process.
  {
    pixtype_t const pixtype = config_value<pixtype_t>(priv::config_pixtype);
    pixfmt_t const pixfmt = config_value<pixfmt_t>(priv::config_pixfmt);
    path_t const path = config_value<path_t>(priv::config_path);
    bool const verify = config_value<bool>(priv::config_verify);
    istream_impl_t const impl = config_value<istream_impl_t>(priv::config_impl);

    istream_t const istr = istream_for_impl(impl, path);

    istream_read_func_t const func = istream_read_for_pixtype(pixtype, pixfmt);

    d.reset(new priv(path, istr, func, verify));
  }

  if (!d->istream)
  {
    static std::string const reason = "The given implementation "
                                      " could not be created";

    throw invalid_configuration_exception(name(), reason);
  }

  if (!d->istream->is_open())
  {
    static std::string const reason = "The input stream is not open";

    throw invalid_configuration_exception(name(), reason);
  }

  if (!d->read)
  {
    static std::string const reason = "A read function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }

  path_t::string_type const path = d->path.native();

  if (path.empty())
  {
    config::value_t const file_path = config::value_t(path.begin(), path.end());
    static std::string const reason = "The path given was empty";

    throw invalid_configuration_value_exception(name(), priv::config_path, file_path, reason);
  }
}

void
video_reader_process
::_init()
{
  if (d->verify)
  {
    if (d->istream->is_seekable())
    {
      bool ok = true;

      while (ok)
      {
        datum_t const dat = d->read(d->istream);

        bool done = false;

        switch (dat->type())
        {
          case datum::empty:
            /// \todo Log that there's a frame that could not be read.
          case datum::data:
            break;
          case datum::flush:
          case datum::complete:
            done = true;
            break;
          case datum::invalid:
          case datum::error:
          default:
            ok = false;
            break;
        }

        if (done)
        {
          break;
        }
      }

      if (!ok)
      {
        static std::string const reason = "The video file has invalid frames in it";

        throw invalid_configuration_exception(name(), reason);
      }

      if (!d->istream->seek_frame(0))
      {
        static std::string const reason = "Unable to seek istream to the "
                                          "beginning after verification";

        throw invalid_configuration_exception(name(), reason);
      }
    }
    else
    {
      /// \todo Log a warning that verification was requested on an unseekable stream.
    }
  }

  process::_init();
}

void
video_reader_process
::_reset()
{
  d->istream->close();

  process::_reset();
}

void
video_reader_process
::_step()
{
  datum_t dat;

  dat = d->read(d->istream);

  push_datum_to_port(priv::port_output, dat);

  if (dat->type() == datum::complete)
  {
    mark_process_as_complete();
  }

  process::_step();
}

video_reader_process::priv
::priv(path_t const& input_path, istream_t const& istr, istream_read_func_t func, bool ver)
  : path(input_path)
  , istream(istr)
  , read(func)
  , verify(ver)
{
}

video_reader_process::priv
::~priv()
{
}

}
