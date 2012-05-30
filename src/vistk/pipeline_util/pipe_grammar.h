/*ckwg +5
 * Copyright 2011-2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_UTIL_PIPE_GRAMMAR_H
#define VISTK_PIPELINE_UTIL_PIPE_GRAMMAR_H

#include "pipeline_util-config.h"

#include "pipe_declaration_types.h"

#include <string>

/**
 * \file pipe_grammar.h
 *
 * \brief Functions to parse pipeline blocks from a string.
 */

namespace vistk
{

/**
 * \brief Parse pipeline blocks from a string.
 *
 * \param str The string to parse.
 *
 * \returns The pipeline blocks within the string.
 */
pipe_blocks VISTK_PIPELINE_UTIL_NO_EXPORT parse_pipe_blocks_from_string(std::string const& str);

}

#endif // VISTK_PIPELINE_UTIL_PIPE_GRAMMAR_H
