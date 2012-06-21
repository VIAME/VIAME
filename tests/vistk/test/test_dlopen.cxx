/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <iostream>
#include <string>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>

int
main(int argc, char* argv[])
{
  if (argc != 3)
  {
    TEST_ERROR("Expected two arguments");

    return EXIT_FAILURE;
  }

#if defined(_WIN32) || defined(_WIN64)
  typedef HMODULE library_t;
  typedef std::wstring module_path_t;
#else
  typedef void* library_t;
  typedef std::string module_path_t;
#endif

  std::string const library = argv[1];
  module_path_t const library_path = argv[2];

  library_t handle = NULL;

#if defined(_WIN32) || defined(_WIN64)
  handle = LoadLibraryW(library_path.c_str());
#else
  handle = dlopen(library_path.c_str(), RTLD_NOW);
#endif

  if (!handle)
  {
    std::string error;

#if defined(_WIN32) || defined(_WIN64)
    {
      DWORD const err_code = GetLastError();
      LPTSTR* str;

      DWORD const ret = FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,
        err_code,
        0,
        static_cast<LPTSTR>(&str),
        0,
        NULL);

      if (!ret)
      {
        TEST_ERROR("Could not get error string from system");

        return EXIT_FAILURE;
      }

      error = str;

      LocalFree(str);
    }
#else
    error = dlerror();
#endif

    TEST_ERROR("Failed to open library " << library << ": " << error);
  }
  else
  {
#if defined(_WIN32) || defined(_WIN64)
    int const ret = FreeLibrary(handle);

    if (!ret)
    {
      std::string error;

      {
        DWORD const err_code = GetLastError();
        LPTSTR* str;

        DWORD const ret = FormatMessage(
          FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
          NULL,
          err_code,
          0,
          static_cast<LPTSTR>(&str),
          0,
          NULL);

        if (!ret)
        {
          TEST_ERROR("Could not get error string from system");

          return EXIT_FAILURE;
        }

        error = str;

        LocalFree(str);
      }

      TEST_ERROR("Failed to close library " << library << ": " << error);
    }
#else
    int const ret = dlclose(handle);

    if (ret)
    {
      std::string const error = dlerror();

      TEST_ERROR("Failed to close library " << library << ": " << error);
    }
#endif
  }

  return EXIT_SUCCESS;
}
