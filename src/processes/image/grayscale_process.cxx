/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "grayscale_process.h"

#include <vistk/pipeline_types/image_types.h>

#include <vistk/pipeline/process_exception.h>

#include <boost/function.hpp>

#include <vil/vil_convert.h>
#include <vil/vil_image_view.h>

namespace vistk
{

class grayscale_process::priv
{
  public:
    typedef boost::function<datum_t (datum_t const&)> convert_func_t;

    priv(config::value_t const& pix, convert_func_t func);
    ~priv();

    config::value_t const pixtype;
    convert_func_t const convert;

    static config::key_t const config_pixtype;
    static config::value_t const default_pixtype;
    static port_t const port_input;
    static port_t const port_output;
};

config::key_t const grayscale_process::priv::config_pixtype = config::key_t("pixtype");
config::value_t const grayscale_process::priv::default_pixtype = config::value_t("byte");
process::port_t const grayscale_process::priv::port_input = port_t("rgbimage");
process::port_t const grayscale_process::priv::port_output = port_t("grayimage");

template<class T>
struct convert
{
  typedef vil_image_view<T> rgb_image_t;
  typedef vil_image_view<T> grayscale_image_t;

  static process::port_type_t const port_type_input;
  static process::port_type_t const port_type_output;

  static datum_t convert_to_gray(datum_t const& dat);
};

template<>
process::port_type_t const convert<uint8_t>::port_type_input = image_types::t_byte_rgb;
template<>
process::port_type_t const convert<uint8_t>::port_type_output = image_types::t_byte_grayscale;

template<>
process::port_type_t const convert<float>::port_type_input = image_types::t_float_rgb;
template<>
process::port_type_t const convert<float>::port_type_output = image_types::t_float_grayscale;

grayscale_process
::grayscale_process(config_t const& config)
  : process(config)
{
  config::value_t pixtype = config->get_value<config::value_t>(priv::config_pixtype, priv::default_pixtype);

  port_type_t port_type_input = type_none;
  port_type_t port_type_output = type_none;

  priv::convert_func_t func = NULL;

  if (pixtype == "byte")
  {
    port_type_input = convert<uint8_t>::port_type_input;
    port_type_output = convert<uint8_t>::port_type_output;

    func = convert<uint8_t>::convert_to_gray;
  }
  else if (pixtype == "float")
  {
    port_type_input = convert<float>::port_type_input;
    port_type_output = convert<float>::port_type_output;

    func = convert<float>::convert_to_gray;
  }

  d = boost::shared_ptr<priv>(new priv(pixtype, func));

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_input, port_info_t(new port_info(
    port_type_input,
    required,
    port_description_t("The image to turn into grayscale."))));
  declare_output_port(priv::port_output, port_info_t(new port_info(
    port_type_output,
    required,
    port_description_t("The resulting grayscale image."))));

  declare_configuration_key(priv::config_pixtype, conf_info_t(new conf_info(
    boost::lexical_cast<config::value_t>(priv::default_pixtype),
    config::description_t("The pixel type of the input images."))));
}

grayscale_process
::~grayscale_process()
{
}

void
grayscale_process
::_init()
{
  if (!d->convert)
  {
    static std::string const reason = "A conversion function for the "
                                      "given pixtype could not be found";

    throw invalid_configuration_exception(name(), reason);
  }
}

void
grayscale_process
::_step()
{
  edge_datum_t const input_dat = grab_from_port(priv::port_input);
  datum_t const input_datum = input_dat.get<0>();
  stamp_t const input_stamp = input_dat.get<1>();

  datum_t dat;

  switch (input_datum->type())
  {
    case datum::DATUM_DATA:
      dat = d->convert(input_datum);
      break;
    case datum::DATUM_EMPTY:
      dat = datum::empty_datum();
      break;
    case datum::DATUM_COMPLETE:
      dat = datum::complete_datum();
      mark_as_complete();
      break;
    case datum::DATUM_ERROR:
      dat = datum::error_datum("Error on the input edges.");
      break;
    case datum::DATUM_INVALID:
    default:
      dat = datum::error_datum("Unrecognized datum type.");
      break;
  }

  edge_datum_t const edat = edge_datum_t(dat, input_stamp);

  push_to_port(priv::port_output, edat);

  process::_step();
}

grayscale_process::priv
::priv(config::value_t const& pix, convert_func_t func)
  : pixtype(pix)
  , convert(func)
{
}

grayscale_process::priv
::~priv()
{
}

template<class T>
datum_t
convert<T>
::convert_to_gray(datum_t const& dat)
{
  rgb_image_t rgb_image = dat->get_datum<rgb_image_t>();

  if (rgb_image.nplanes() != 3)
  {
    return datum::error_datum("Input image does not have three planes.");
  }

  grayscale_image_t gray_image;

  vil_convert_planes_to_grey(rgb_image, gray_image);

  return datum::new_datum(gray_image);
}

}
