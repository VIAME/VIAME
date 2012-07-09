/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_TOOLS_HELPERS_MAIN_H
#define VISTK_TOOLS_HELPERS_MAIN_H

static int vistk_main(int argc, char* argv[]);

#define tool_main                                               \
main(int argc, char* argv[])                                    \
{                                                               \
  try                                                           \
  {                                                             \
    return vistk_main(argc, argv);                              \
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
vistk_main

#endif // VISTK_TOOLS_HELPERS_MAIN_H
