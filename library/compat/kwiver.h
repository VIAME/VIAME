// This file is part of VIAME, and is distributed under an OSI-approved
// BSD 3-Clause License. See either the root top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/**
 * \file
 * \brief The namespace names VIAME had before phase 11, for one release.
 *
 * Open decision 8: `kwiver::vital` became `viame`, `kwiver::arrows::<a>`
 * became `viame::<a>`, and `sprokit` became `viame::pipeline`. Code outside
 * VIAME that still writes the old names includes this header and keeps
 * compiling:
 *
 * \code
 * #include <viame/compat/kwiver.h>
 *
 * kwiver::vital::image_container_sptr image;   // viame::image_container_sptr
 * sprokit::process_t proc;                     // viame::pipeline::process_t
 * \endcode
 *
 * These are namespace aliases, so they name the same types -- there is no
 * second set of anything, and a `viame::image` and a `kwiver::vital::image`
 * are one type, not two that convert.
 *
 * This header ships in the release phase 11 lands in and is removed in the
 * next one. Nothing inside VIAME includes it.
 */

#ifndef VIAME_COMPAT_KWIVER_H
#define VIAME_COMPAT_KWIVER_H

namespace viame {

// Declared before they are aliased, so that including this header first still
// works.
namespace pipeline {}
namespace tools {}

// `kwiver::vital::x` and `kwiver::arrows::ocv::x` reach `viame::x` through
// `kwiver` below; these are what make the middle name disappear.
namespace vital = ::viame;
namespace arrows = ::viame;

// `kwiver::sprokit::python`, which the python bindings used.
namespace sprokit = ::viame::pipeline;

} // namespace viame

// The old top level names.
namespace kwiver = ::viame;
namespace sprokit = ::viame::pipeline;

#endif // VIAME_COMPAT_KWIVER_H
