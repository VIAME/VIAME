/*ckwg +29
 * Copyright 2011-2012 by Kitware, Inc.
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

#ifndef SPROKIT_PIPELINE_UTIL_PIPE_GRAMMAR_H
#define SPROKIT_PIPELINE_UTIL_PIPE_GRAMMAR_H

#include "pipeline_util-config.h"

#include "pipe_declaration_types.h"

#include <string>

/**
 * \file pipe_grammar.h
 *
 * \brief Functions to parse pipeline blocks from a string.
 */

namespace sprokit
{

/**
 * \brief Parse pipeline blocks from a string.
 *
 * \param str The string to parse.
 *
 * \returns The pipeline blocks within the string.
 */
SPROKIT_PIPELINE_UTIL_NO_EXPORT pipe_blocks parse_pipe_blocks_from_string(std::string const& str);

/**
 * \brief Parse cluster blocks from a string.
 *
 * \param str The string to parse.
 *
 * \returns The cluster blocks within the string.
 */
cluster_blocks SPROKIT_PIPELINE_UTIL_NO_EXPORT parse_cluster_blocks_from_string(std::string const& str);

}

#endif // SPROKIT_PIPELINE_UTIL_PIPE_GRAMMAR_H
