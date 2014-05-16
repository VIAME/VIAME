/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_KW_ARCHIVE_WRITER_PROCESS_H_
#define _KWIVER_KW_ARCHIVE_WRITER_PROCESS_H_

#include <sprokit/pipeline/process.h>

#include <boost/scoped_ptr.hpp>
#include <maptk/core/vector.h>

namespace kwiver
{

/*!
 * \class kw_archive_index_writer
 *
 * \brief KW archive writer process
 *
 * \process Writes kw video archive
 *
 * \iports
 *
 * \iport{timestamp}
 */

class kw_archive_writer_process
  : public sprokit::process
{
public:
  // -- TYPES --
  //+ would be nice to have accessors - float lat(); float lon();
  typedef maptk::vector_2_ < float > lat_lon_t; // remember, that's (y,x)

  // points are ordered ul, ur. lr, ll (lat, lon)
  //+ would be nice to have accessors ul(); ur(); ...
  typedef maptk::vector_ < 4, lat_lon_t > corner_points_t;
  static sprokit::process::type_t const kwiver_corner_points;

  /**
   * \brief Constructor
   *
   * @param config Process config object
   *
   * @return
   */
  kw_archive_writer_process(sprokit::config_t const& config);

  /**
   * \brief Destructor
   */
  ~kw_archive_writer_process();

protected:
  void _configure();
  void _init();
  void _step();

private:
  void make_ports();
  void make_config();

  class priv;
  boost::scoped_ptr<priv> d;
};

} // end namespace kwiver

#endif /* _KWIVER_KW_ARCHIVE_WRITER_PROCESS_H_ */
