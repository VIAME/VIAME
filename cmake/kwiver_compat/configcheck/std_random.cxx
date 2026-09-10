#include <random>
typedef std::normal_distribution<> norm_dist_t;

int
main( int argc, char* argv[] )
{
  std::mt19937 rng;
  norm_dist_t norm( 0.0, 1.2 );
  double val = norm( rng );
  return 0;
}
