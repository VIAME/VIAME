/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */


#ifndef _KWIVER_CONFIG_UTIL_H_
#define _KWIVER_CONFIG_UTIL_H_

#include <sprokit/pipeline/config.h>
#include <maptk/core/config_block.h>


namespace kwiver
{

  /**
   * \brief Convert from sprokit to maptk config.
   *
   * Convert sprokit config block to maptk config block. All the
   * entries (key, value) in the \b from config block are added to the
   * \b to config block.  Existing entries in the \b to config are
   * overwritten unless they are marked as read_only. They are not
   * modified in that case.
   *
   * It is hoped that this approach is short lived and that both
   * sprokit and maptk will use the same config class.
   *
   * @param from sprokit config block for source entries
   * @param to maptk config block to get new entries
   */
  void convert_config( sprokit::config_t const  from,
                       maptk::config_block_sptr to );

  /**
   * \brief Convert from maptk config to sprokit config.
   *
   * Convert maptk config block to pprokit config block. All the
   * entries (key, value) in the \b from config block are added to the
   * \b to config block.  Existing entries in the \b to config are
   * overwritten unless they are marked as read_only. They are not
   * modified in that case.
   *
   * It is hoped that this approach is short lived and that both
   * sprokit and maptk will use the same config class.
   *
   * @param from maptk config block
   * @param to sprokit config block to get new entries
   */
  void convert_config( maptk::config_block_sptr const from,
                       sprokit::config_t              to );


} //end namespace

#endif /* _KWIVER_CONFIG_UTIL_H_ */
