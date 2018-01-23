/*ckwg +29
 * Copyright 2012 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SPROKIT_PIPELINE_VERSION_H
#define SPROKIT_PIPELINE_VERSION_H

#include <sprokit/pipeline/sprokit_pipeline_export.h>

#include <string>

/**
 * \file version.h
 *
 * \brief Runtime version checks.
 */

namespace sprokit
{

/**
 * \class version "version.h" <sprokit/pipeline/version.h>
 *
 * \brief Runtime version information.
 */
class SPROKIT_PIPELINE_EXPORT version
{
  public:
    /// The type of version components.
    typedef uint64_t version_t;

    /// The major version number.
    static version_t const major;
    /// The minor version number.
    static version_t const minor;
    /// The patch version number.
    static version_t const patch;
    /// The version string.
    static std::string const version_string;

    /// True if information from the git repository is available, false otherwise.
    static bool const git_build;
    /// The full git hash of the build.
    static std::string const git_hash;
    /// An abbreviated git hash of the build.
    static std::string const git_hash_short;
    /// Empty if the git repository was clean at the time of the build.
    static std::string const git_dirty;

    /**
     * \brief Check whether the runtime version is at least some value.
     *
     * \param major_ The major version to test.
     * \param minor_ The minor version to test.
     * \param patch_ The patch version to test.
     *
     * \returns True if the runtime version is at least the given version.
     */
    static bool check(version_t major_, version_t minor_, version_t patch_);
};

}

#endif // SPROKIT_PIPELINE_VERSION_H
