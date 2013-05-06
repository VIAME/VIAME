/*ckwg +5
 * Copyright 2012-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_TOOLS_HELPERS_TOOL_MAIN_H
#define SPROKIT_TOOLS_HELPERS_TOOL_MAIN_H

static int sprokit_main(int argc, char* argv[]);

#define tool_main                                               \
main(int argc, char* argv[])                                    \
{                                                               \
  try                                                           \
  {                                                             \
    return sprokit_main(argc, argv);                              \
  }                                                             \
  catch (std::exception const& e)                               \
  {                                                             \
    std::cerr << "Exception caught: " << e.what() << std::endl; \
                                                                \
    return EXIT_FAILURE;                                        \
  }                                                             \
  catch (...)                                                   \
  {                                                             \
    std::cerr << "Unknown exception caught" << std::endl;       \
                                                                \
    return EXIT_FAILURE;                                        \
  }                                                             \
}                                                               \
                                                                \
int                                                             \
sprokit_main

#endif // SPROKIT_TOOLS_HELPERS_TOOL_MAIN_H
