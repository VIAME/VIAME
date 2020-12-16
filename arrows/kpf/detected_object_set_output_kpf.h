// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface for detected_object_set_output_kpf
 */

#ifndef KWIVER_ARROWS_DETECTED_OBJECT_SET_OUTPUT_KPF_H
#define KWIVER_ARROWS_DETECTED_OBJECT_SET_OUTPUT_KPF_H

#include <arrows/kpf/kwiver_algo_kpf_export.h>

#include <vital/algo/detected_object_set_output.h>

namespace kwiver {
namespace arrows {
namespace kpf {

class KWIVER_ALGO_KPF_EXPORT detected_object_set_output_kpf
  : public vital::algo::detected_object_set_output
{
public:
  detected_object_set_output_kpf();
  virtual ~detected_object_set_output_kpf();

  virtual vital::config_block_sptr get_configuration() const;
  virtual void set_configuration( vital::config_block_sptr config );
  virtual bool check_configuration( vital::config_block_sptr config ) const;

  virtual void write_set( const kwiver::vital::detected_object_set_sptr set, std::string const& image_name );

private:
  class priv;
  std::unique_ptr< priv > d;
};

} } } // end namespace

#endif // KWIVER_ARROWS_DETECTED_OBJECT_SET_OUTPUT_KPF_H
