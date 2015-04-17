/*ckwg +5
 * Copyright 2014 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef _KWIVER_KWIVER_EXCEPTIONS_H_
#define _KWIVER_KWIVER_EXCEPTIONS_H_

#include <string>
#include <sstream>

class kwiver_base_exception
{
public:
  kwiver_base_exception( char const* file, int line, std::string const& w )
    : m_what(w), m_file(file), m_line(line) { }

  kwiver_base_exception( std::string const& w ) : m_what(w), m_line(-1) { }

private:
  std::string m_what;

  // location where exception was thrown
  std::string m_file;
  int m_line;
};


/**
 * \brief Exception helper macro.
 *
 * Macro to simplify creating exception messages using stream
 * operators.
 *
 * @param E Exception type.
 * @param MSG stream constructed exception message.
 */
#define KWIVER_THROW(E, MSG) do {               \
    std::stringstream _oss_;                    \
  _oss_ << MSG;                                 \
  throw E( __file, __line, MSG.str() );         \
} while (0)

#endif /* _KWIVER_KWIVER_EXCEPTIONS_H_ */
