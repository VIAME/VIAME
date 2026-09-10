#include <cstddef>

int
foo( std::nullptr_t )
{
  return 0;
}

int
main()
{
  return foo( nullptr );
}
