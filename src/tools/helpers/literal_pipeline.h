/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_TOOLS_HELPERS_LITERAL_PIPELINE_H
#define SPROKIT_TOOLS_HELPERS_LITERAL_PIPELINE_H

#define PROCESS(type, name) \
  "process " name "\n"      \
  "  :: " type "\n"

#define FLAGS_WRAP(flags) \
  "[" flags "]"

#define PROVIDER_WRAP(provider) \
  "{" provider "}"

#define CONFIG_RAW(key, flags, provider, value) \
  "  :" key flags provider " " value "\n"

#define CONFIG_FULL(key, flags, provider, value) \
  CONFIG_RAW(key, FLAGS_WRAP(flags), PROVIDER_WRAP(provider), value)

#define CONFIG_FLAGS(key, flags, value) \
  CONFIG_RAW(key, FLAGS_WRAP(flags), "", value)

#define CONFIG_PROVIDER(key, provider, value) \
  CONFIG_RAW(key, "", PROVIDER_WRAP(provider), value)

#define CONFIG(key, value) \
  CONFIG_RAW(key, "", "", value)

#define CONFIG_BLOCK(name) \
  "config " name "\n"

#define INPUT_MAPPING(port, mapped, mapped_port) \
  "  imap from " port "\n"                       \
  "       to   " mapped "." mapped_port "\n"

#define OUTPUT_MAPPING(port, mapped, mapped_port) \
  "  omap from " port "\n"                        \
  "       to   " mapped "." mapped_port "\n"

#define CONNECT(up_name, up_port, down_name, down_port) \
  "connect from " up_name "." up_port "\n"              \
  "        to   " down_name "." down_port "\n"

#endif // SPROKIT_TOOLS_HELPERS_LITERAL_PIPELINE_H
