// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <test_common.h>

#include <sprokit/pipeline_util/path.h>

#if defined(_WIN32) || defined(_WIN64)
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

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

#if defined(_WIN32) || defined(_WIN64)
  typedef HMODULE library_t;
#else
  typedef void* library_t;
#endif

  std::string const library = argv[1];
  sprokit::path_t const path = argv[2];

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
