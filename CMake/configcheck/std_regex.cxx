#include <regex>
int main(int argc, char *argv[])
{
  std::basic_regex<char> integer_pattern
    ("(-)?(0x)?([0-9a-zA-Z]+)|((0x)?0)");

  return 0;
}
