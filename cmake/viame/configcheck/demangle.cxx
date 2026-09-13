#include <cxxabi.h>
#include <string>

int
main()
{
  const char* sym = "printf";
  std::string tname;
  int status;
  char* demangled_name = abi::__cxa_demangle( sym, NULL, NULL, &status );

  return 0;
}
