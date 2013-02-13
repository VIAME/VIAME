/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <test_common.h>

#include <vistk/utilities/path.h>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdlib>

#if defined(_WIN32) || defined(_WIN64)
namespace
{

typedef std::basic_string<TCHAR> tstring;

}

static tstring last_windows_error();
#endif

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

#if defined(_WIN32) || defined(_WIN64)
  typedef HMODULE library_t;
#else
  typedef void* library_t;
#endif
  typedef vistk::path_t module_path_t;

  std::string const library = argv[1];
  vistk::path_t const path = argv[2];

  library_t handle = NULL;

#if defined(_WIN32) || defined(_WIN64)
  handle = LoadLibraryW(path.native().c_str());
#else
  handle = dlopen(path.native().c_str(), RTLD_NOW);
#endif

  if (!handle)
  {
#if defined(_WIN32) || defined(_WIN64)
    tstring const error = last_windows_error();
#else
    std::string const error = dlerror();
#endif

    TEST_ERROR("Failed to open library " << library << ": " << error);
  }
  else
  {
#if defined(_WIN32) || defined(_WIN64)
    int const ret = FreeLibrary(handle);

    if (!ret)
    {
      tstring const error = last_windows_error();
#else
    int const ret = dlclose(handle);

    if (ret)
    {
      std::string const error = dlerror();
#endif

      TEST_ERROR("Failed to close library " << library << ": " << error);
    }
  }

  return EXIT_SUCCESS;
}

#if defined(_WIN32) || defined(_WIN64)
tstring
last_windows_error()
{
  DWORD const err_code = GetLastError();
  LPTSTR str;

  DWORD const ret = FormatMessage(
    FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
    NULL,
    err_code,
    0,
    reinterpret_cast<LPTSTR>(&str),
    0,
    NULL);

  if (!ret)
  {
    TEST_ERROR("Could not get error string from system");

    exit(EXIT_FAILURE);
  }

  tstring const error = str;

  LocalFree(str);

  return error;
}
#endif
