/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
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
