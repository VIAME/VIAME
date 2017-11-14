#include <yaml-cpp/yaml.h>
#include <iostream>
#include <fstream>
#include <string>

using std::cout;
using std::cerr;
using std::ostream;
using std::ifstream;
using std::string;

string
indent( size_t depth )
{
  string ret;
  for (size_t i=0; i<depth; ++i)
  {
    ret += "  ";
  }
  return ret;
}

void visit( ostream& os, size_t depth, const YAML::Node& n )
{
  os << indent( depth );
  switch (n.Type())
  {
  case YAML::NodeType::Null:
    os << "(null)\n";
    break;
  case YAML::NodeType::Scalar:
    os << "(scalar) '" << n.as<string>() << "'\n";
    break;
  case YAML::NodeType::Sequence:
    os << "(sequence)\n";
    for (auto it=n.begin(); it != n.end(); ++it)
    {
      visit( os, depth+1, *it);
    }
    break;
  case YAML::NodeType::Map:
    cerr << "(map)\n";
    for (auto it=n.begin(); it != n.end(); ++it)
    {
      os << indent(depth+1) << it->first.as<string>() << " => \n";
      visit( os, depth+2, it->second);
    }
    break;
  case YAML::NodeType::Undefined:
    os << "(undef)\n";
    break;
  default:
    os << "(unhandled " << n.Type() << ")\n";
    break;
  }
}

int main( int argc, char *argv[] )
{
  if (argc != 2)
  {
    cerr << "Usage: " << argv[0] << " file.yml\n";
    return EXIT_FAILURE;
  }

  try
  {
    ifstream is( argv[1] );
    YAML::Node doc = YAML::Load( is );

    for (auto it = doc.begin(); it != doc.end(); ++it)
    {
      visit( cout, 0, *it);
    }
  }
  catch (const YAML::Exception& e )
  {
    cerr << "YAML exception: " << e.what() << "\n";
  }

}
