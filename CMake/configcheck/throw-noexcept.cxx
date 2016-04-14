void f() noexcept; // the function f() does not throw
void (*fp)() noexcept(false); // fp points to a function that may throwint main()

int main()
{
  return 0;
}
