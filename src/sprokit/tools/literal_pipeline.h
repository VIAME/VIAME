/*ckwg +29
 * Copyright 2012-2013 by Kitware, Inc.
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

#ifndef SPROKIT_TOOLS_LITERAL_PIPELINE_H
#define SPROKIT_TOOLS_LITERAL_PIPELINE_H

#define SPROKIT_PROCESS(type, name) \
  "process " name "\n"              \
  "  :: " type "\n"

#define SPROKIT_FLAGS_WRAP(flags) \
  "[" flags "]"

#define SPROKIT_PROVIDER_WRAP(provider) \
  "{" provider "}"

#define SPROKIT_CONFIG_RAW(key, flags, provider, value) \
  "  :" key flags provider " " value "\n"

#define SPROKIT_CONFIG_FULL(key, flags, provider, value) \
  SPROKIT_CONFIG_RAW(key, SPROKIT_FLAGS_WRAP(flags), SPROKIT_PROVIDER_WRAP(provider), value)

#define SPROKIT_CONFIG_FLAGS(key, flags, value) \
  SPROKIT_CONFIG_RAW(key, SPROKIT_FLAGS_WRAP(flags), "", value)

#define SPROKIT_CONFIG_PROVIDER(key, provider, value) \
  SPROKIT_CONFIG_RAW(key, "", SPROKIT_PROVIDER_WRAP(provider), value)

#define SPROKIT_CONFIG(key, value) \
  SPROKIT_CONFIG_RAW(key, "", "", value)

#define SPROKIT_CONFIG_BLOCK(name) \
  "config " name "\n"

#define SPROKIT_CONNECT(up_name, up_port, down_name, down_port) \
  "connect from " up_name "." up_port "\n"                      \
  "        to   " down_name "." down_port "\n"

#endif // SPROKIT_TOOLS_LITERAL_PIPELINE_H
