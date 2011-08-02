/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H
#define VISTK_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H

#include "pipeline_util-config.h"

#include <vistk/pipeline/types.h>

#include <boost/filesystem/path.hpp>

#include <string>

/**
 * \file load_pipe_exception.h
 *
 * \brief Header for exceptions used when loading a pipe declaration.
 */

namespace vistk
{

/**
 * \class load_pipe_exception load_pipe_exception.h <vistk/pipeline_util/load_pipe_exception.h>
 *
 * \brief The base class for all exceptions thrown when loading a pipe declaration.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_UTIL_EXPORT load_pipe_exception
  : public pipeline_exception
{
};

/**
 * \class file_no_exist_exception load_pipe_exception.h <vistk/pipeline_util/load_pipe_exception.h>
 *
 * \brief The exception thrown when a file does not exist.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_UTIL_EXPORT file_no_exist_exception
  : public load_pipe_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param fname The path that does not exist.
     */
    file_no_exist_exception(boost::filesystem::path const& fname) throw();
    /**
     * \brief Destructor.
     */
    ~file_no_exist_exception() throw();

    /// The path that does not exist.
    boost::filesystem::path const m_fname;

    /**
     * \brief A description of the exception.
     *
     * \returns A string describing what went wrong.
     */
    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class file_open_exception load_pipe_exception.h <vistk/pipeline_util/load_pipe_exception.h>
 *
 * \brief The exception thrown when a file could not be opened.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_UTIL_EXPORT file_open_exception
  : public load_pipe_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param fname The path that was unable to be loaded.
     */
    file_open_exception(boost::filesystem::path const& fname) throw();
    /**
     * \brief Destructor.
     */
    ~file_open_exception() throw();

    /// The path that was unable to be loaded.
    boost::filesystem::path const m_fname;

    /**
     * \brief A description of the exception.
     *
     * \returns A string describing what went wrong.
     */
    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class stream_failure_exception load_pipe_exception.h <vistk/pipeline_util/load_pipe_exception.h>
 *
 * \brief The exception thrown when a stream has an error.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_UTIL_EXPORT stream_failure_exception
  : public load_pipe_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param msg The message given for the stream failure.
     */
    stream_failure_exception(std::string const& msg) throw();
    /**
     * \brief Destructor.
     */
    ~stream_failure_exception() throw();

    /// The message given for the stream failure.
    std::string const m_msg;

    /**
     * \brief A description of the exception.
     *
     * \returns A string describing what went wrong.
     */
    char const* what() const throw();
  private:
    std::string m_what;
};

/**
 * \class failed_to_parse load_pipe_exception.h <vistk/pipeline_util/load_pipe_exception.h>
 *
 * \brief The exception thrown when a parse error occurred.
 *
 * \ingroup exceptions
 */
class VISTK_PIPELINE_UTIL_EXPORT failed_to_parse
  : public load_pipe_exception
{
  public:
    /**
     * \brief Constructor.
     *
     * \param reason A reason for the failure
     * \param where Where the error occurred.
     */
    failed_to_parse(std::string const& reason, std::string const& where) throw();
    /**
     * \brief Destructor.
     */
    ~failed_to_parse() throw();

    /// The reason for the failure to parse.
    std::string const m_reason;
    /// Where the error occurred.
    std::string const m_where_full;
    /// Where the error occurred, abbreviated to 64 bytes.
    std::string const m_where_brief;

    /**
     * \brief A description of the exception.
     *
     * \returns A string describing what went wrong.
     */
    char const* what() const throw();
  private:
    static size_t const max_size;

    std::string m_what;
};

} // end namespace vistk

#endif // VISTK_PIPELINE_UTIL_LOAD_PIPE_EXCEPTION_H
