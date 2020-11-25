// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Interface to matlab exceptions
 */

#ifndef KWIVER_VITAL_MATLAB_EXCEPTION_H
#define KWIVER_VITAL_MATLAB_EXCEPTION_H

#include <vital/vital_config.h>
#include <arrows/matlab/kwiver_algo_matlab_export.h>
#include <vital/exceptions/base.h>

namespace kwiver {
namespace arrows {
namespace matlab {

// -----------------------------------------------------------------
/**
 *
 *
 */
class KWIVER_ALGO_MATLAB_EXPORT matlab_exception
  : public vital::vital_exception
{
public:
  // -- CONSTRUCTORS --
  matlab_exception( const std::string& msg ) noexcept;

  virtual ~matlab_exception() noexcept;

}; // end class matlab_exception

} } } // end namespace

#endif /* KWIVER_VITAL_MATLAB_EXCEPTION_H */
