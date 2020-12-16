// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 *
 * \brief Supplemental macro definitions for test cases
 */

#ifndef KWIVER_TEST_TEST_TMPFN_H_
#define KWIVER_TEST_TEST_TMPFN_H_

#include <string>

#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#define tempnam(d, p) _tempnam(d, p)
#endif

namespace kwiver {
namespace testing {

// ----------------------------------------------------------------------------
/** @brief Generate a unique file name in the current working directory.
 *
 * @param prefix Prefix for generated file name.
 * @param suffix Suffix for generated file name.
 */
std::string
temp_file_name( char const* prefix, char const* suffix )
{
  auto const n = tempnam(".", prefix);
  auto const s = std::string(n);
  free(n);

  return s + suffix;
}

} // end namespace testing
} // end namespace kwiver

#endif
