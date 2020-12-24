// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file load_pipe_exception.h
 *
 * \brief Header for exceptions used when loading a pipe declaration.
 */

#ifndef SPROKIT_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H
#define SPROKIT_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H

#include<sprokit/pipeline_util/sprokit_pipeline_util_export.h>

#include <vital/vital_types.h>

#include <sprokit/pipeline/types.h>

#include <string>
#include <cstddef>

namespace sprokit {

/**
 * \class load_pipe_exception load_pipe_exception.h <sprokit/pipeline_util/load_pipe_exception.h>
 *
 * \brief The base class for all exceptions thrown when loading a pipe declaration.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT load_pipe_exception
  : public pipeline_exception
{
  public:
    /**
     * \brief Constructor.
     */
    load_pipe_exception() noexcept;

    /**
     * \brief Destructor.
     */
    virtual ~load_pipe_exception() noexcept;
};

/**
 * \class file_no_exist_exception load_pipe_exception.h <sprokit/pipeline_util/load_pipe_exception.h>
 *
 * \brief The exception thrown when a file does not exist.
 *
 * \ingroup exceptions
 */
class SPROKIT_PIPELINE_UTIL_EXPORT file_no_exist_exception
  : public load_pipe_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param fname The path that does not exist.
     */
  file_no_exist_exception(kwiver::vital::path_t const& fname) noexcept;
    /**
     * \brief Destructor.
     */
    virtual ~file_no_exist_exception() noexcept;

    /// The path that does not exist.
    kwiver::vital::path_t const m_fname;
};

// ------------------------------------------------------------------
class SPROKIT_PIPELINE_UTIL_EXPORT parsing_exception
  : public load_pipe_exception
{
public:
  /**
   * \brief Constructor.
   */
  parsing_exception( const std::string& msg) noexcept;

  /**
   * \brief Destructor.
   */
  virtual ~parsing_exception() noexcept;

};

}

#endif // SPROKIT_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H
