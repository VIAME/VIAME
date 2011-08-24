/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "grayscale_process.h"

#include <vistk/pipeline_types/image_types.h>

#include <vil/vil_convert.h>
#include <vil/vil_image_view.h>

namespace vistk
{

template<class T>
class grayscale_process<T>::priv
{
  public:
    priv();
    ~priv();

    typedef vil_image_view<T> rgb_image_t;
    typedef vil_image_view<T> grayscale_image_t;

    static port_type_t const port_type_input;
    static port_type_t const port_type_output;
    static port_t const port_input;
    static port_t const port_output;
};

template<>
process::port_type_t const grayscale_process<uint8_t>::priv::port_type_input = image_types::t_byte_rgb;
template<>
process::port_type_t const grayscale_process<uint8_t>::priv::port_type_output = image_types::t_byte_grayscale;
template<>
process::port_type_t const grayscale_process<float>::priv::port_type_input = image_types::t_float_rgb;
template<>
process::port_type_t const grayscale_process<float>::priv::port_type_output = image_types::t_float_grayscale;

template<class T>
process::port_t const grayscale_process<T>::priv::port_input = port_t("rgbimage");
template<class T>
process::port_t const grayscale_process<T>::priv::port_output = port_t("grayimage");

template<class T>
grayscale_process<T>
::grayscale_process(config_t const& config)
  : process(config)
{
  d = boost::shared_ptr<priv>(new priv);

  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(priv::port_input, port_info_t(new port_info(
    priv::port_type_input,
    required,
    port_description_t("The image to turn into grayscale."))));
  declare_output_port(priv::port_output, port_info_t(new port_info(
    priv::port_type_output,
    required,
    port_description_t("The resulting grayscale image."))));
}

template<class T>
grayscale_process<T>
::~grayscale_process()
{
}

template<class T>
void
grayscale_process<T>
::_step()
{
  edge_datum_t const input_dat = grab_from_port(priv::port_input);
  datum_t const input_datum = input_dat.get<0>();
  stamp_t const input_stamp = input_dat.get<1>();

  datum_t dat;

  switch (input_datum->type())
  {
    case datum::DATUM_DATA:
    {
      typename priv::rgb_image_t rgb_image = input_datum->get_datum<typename priv::rgb_image_t>();

      if (rgb_image.nplanes() != 3)
      {
        dat = datum::error_datum("Input image does not have three planes.");

        break;
      }

      typename priv::grayscale_image_t gray_image;

      vil_convert_planes_to_grey(rgb_image, gray_image);

      dat = datum::new_datum(gray_image);

      break;
    }
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

template<class T>
grayscale_process<T>::priv
::priv()
{
}

template<class T>
grayscale_process<T>::priv
::~priv()
{
}

process_t
create_grayscale_byte_process(config_t const& config)
{
  return process_t(new grayscale_process<uint8_t>(config));
}

process_t
create_grayscale_float_process(config_t const& config)
{
  return process_t(new grayscale_process<float>(config));
}

}
