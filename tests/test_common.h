/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_TEST_TEST_COMMON_H
#define VISTK_TEST_TEST_COMMON_H

#define EXPECT_EXCEPTION(exc, code, action)        \
  do                                               \
  {                                                \
    bool got_exception = false;                    \
                                                   \
    try                                            \
    {                                              \
      code;                                        \
    }                                              \
    catch (exc& e)                                 \
    {                                              \
      got_exception = true;                        \
                                                   \
      (void)e.what();                              \
    }                                              \
    catch (std::exception& e)                      \
    {                                              \
      std::cerr << "Error: Unexpected exception: " \
                << e.what() << std::endl;          \
                                                   \
      got_exception = true;                        \
    }                                              \
                                                   \
    if (!got_exception)                            \
    {                                              \
      std::cerr << "Error: Did not get "           \
                   "expected exception when "      \
                << action << std::endl;            \
    }                                              \
  } while (false)

#endif // VISTK_TEST_TEST_COMMON_H
