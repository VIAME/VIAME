#include <iostream>

#include <track_oracle/track_field.h>
#include <track_oracle/data_terms/data_terms.h>

using namespace kwiver::track_oracle;

int main(int, char *[])
{
  track_field< dt::tracking::bounding_box > bb;
  std::cout << bb( 0 ) << std::endl;
}
