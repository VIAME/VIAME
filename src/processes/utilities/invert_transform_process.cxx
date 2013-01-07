/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "invert_transform_process.h"

#include <vistk/utilities/homography.h>

#include <vnl/vnl_inverse.h>

/**
 * \file invert_transform_process.cxx
 *
 * \brief Implementation of the invert transform process.
 */

namespace vistk
{

class invert_transform_process::priv
{
  public:
    priv();
    ~priv();

    static port_t const port_input;
    static port_t const port_output;
};

process::port_t const invert_transform_process::priv::port_input = port_t("transform");
process::port_t const invert_transform_process::priv::port_output = port_t("inv_transform");

invert_transform_process
::invert_transform_process(config_t const& config)
  : process(config)
  , d(new priv)
{
  port_flags_t required;

  required.insert(flag_required);

  declare_input_port(
    priv::port_input,
    "transform",
    required,
    port_description_t("The transform to invert."));

  declare_output_port(
    priv::port_output,
    "transform",
    required,
    port_description_t("The inverted transform."));
}

invert_transform_process
::~invert_transform_process()
{
}

void
invert_transform_process
::_step()
{
  typedef homography_base::transform_t transform_t;
  typedef vnl_matrix_fixed<double, 3, 3> matrix_t;

  transform_t const trans = grab_from_port_as<transform_t>(priv::port_input);
  matrix_t const trans_mat = trans.get_matrix();

  transform_t const inv_trans = vnl_inverse(trans_mat);

  push_to_port_as(priv::port_output, inv_trans);

  process::_step();
}

invert_transform_process::priv
::priv()
{
}

invert_transform_process::priv
::~priv()
{
}

}
