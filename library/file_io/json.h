// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/// \file
/// \brief rapidjson, configured the way cereal configured it.
///
/// Three things about how VIAME parses and writes JSON are not rapidjson's
/// defaults, and until P8-T06 all three arrived by accident: the files that
/// use rapidjson included `<cereal/archives/json.hpp>` first, which set them
/// on the way past. Two of the three included it for no other reason -- they
/// call no cereal archive at all -- and a comment in each said so.
///
/// That worked, and it would have gone on working right up until someone
/// removed an include that looked decorative. So the three settings are
/// here, stated, and this header is the only place rapidjson is included
/// from:
///
///  * **an internal assertion throws** rather than calling `assert`. Under
///    `NDEBUG` -- which is every build that ships -- `assert` is nothing at
///    all, so the alternative to throwing is not a crash but undefined
///    behaviour on malformed input;
///  * **NaN and infinity are written** rather than refused. A camera
///    intrinsic that came out NaN reaches the file as `NaN`, which is at
///    least a thing the next reader can see;
///  * **parsing is full precision** and accepts `NaN`, `Infinity` and
///    `-Infinity`. Full precision costs something per number and buys the
///    guarantee that a `double` written by VIAME reads back bit for bit,
///    which is what the calibration goldens are held to.
///
/// Including a rapidjson header directly gets rapidjson's defaults instead.
/// Three files in `plugins/core` do exactly that, and did before P8-T06 as
/// well -- see open question 2.11. Which of the two a file wants is a real
/// question with a real answer, and the point of this header is that it now
/// has to be asked out loud.

#ifndef VIAME_FILE_IO_JSON_H_
#define VIAME_FILE_IO_JSON_H_

#include <stdexcept>
#include <string>

namespace viame {

/// Thrown where rapidjson would have asserted.
///
/// An internal assertion means the document is not the shape the call
/// assumed -- asking an array for a member, say. It is a programming error
/// rather than a data error, which is why it is not `vital::invalid_data`.
struct json_internal_error : std::runtime_error
{
  explicit json_internal_error( char const* what )
    : std::runtime_error( what )
  {}
};

} // namespace viame

// These three have to be set before rapidjson's own headers are read, which
// is the whole reason this file exists.
#ifndef RAPIDJSON_ASSERT
#define RAPIDJSON_ASSERT( x ) if( !( x ) ) {                                 \
  throw ::viame::json_internal_error(                                        \
    "rapidjson internal assertion failure: " #x ); }
#endif

#define RAPIDJSON_WRITE_DEFAULT_FLAGS kWriteNanAndInfFlag
#define RAPIDJSON_PARSE_DEFAULT_FLAGS \
  kParseFullPrecisionFlag | kParseNanAndInfFlag

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>

#endif // VIAME_FILE_IO_JSON_H_
