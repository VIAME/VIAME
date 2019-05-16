#include <regex>
#include <iostream>

int main(int argc, char *argv[])
{
  std::basic_regex<char> integer_pattern
    ("(-)?(0x)?([0-9a-zA-Z]+)|((0x)?0)");
  std::basic_regex<char> truthy_pattern
    ("(t|T)(rue)?");
  std::basic_regex<char> falsy_pattern
    ("((f|F)(alse)?)?");
  std::basic_regex<char> option_matcher
    ("--([[:alnum:]][-_[:alnum:]]+)(=(.*))?|-([[:alnum:]]+)");
  std::basic_regex<char> option_specifier
    ("(([[:alnum:]]),)?[ ]*([[:alnum:]][-_[:alnum:]]*)?");

  std::string text = "12345.123";
  std::smatch match;
  std::regex_match(text, match, integer_pattern);
  std::regex_match(text, match, truthy_pattern);
  std::regex_match(text, match, falsy_pattern);
  std::regex_match(text, match, option_specifier);
  std::regex_match(text, match, option_matcher);
  return 0;
}
