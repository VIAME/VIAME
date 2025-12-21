#include <set>
#include "ocv_kmedians.h"


inline bool
are_centers_different( const cv::Mat& c1, const cv::Mat& c2, double tol = 1e-9 )
{
  if( c1.rows != c2.rows || c1.cols != c2.cols )
    return true;


  for( int i_row = 0; i_row < c1.rows; i_row++ )
    for( int i_col = 0; i_col < c1.cols; i_col++ )
      if( std::abs( c1.at< float >( i_row, i_col ) - c2.at< float >( i_row, i_col ) ) > tol )
        return true;

  return false;
}

inline double
manhatan_distance( const cv::Mat& data, const cv::Mat& centers, int i_data, int i_cluster )
{
  double center_dist{};
  for( int i_feature = 0; i_feature < data.cols; i_feature++ )
    center_dist += std::abs( data.at< float >( i_data, i_feature ) - centers.at< float >( i_cluster, i_feature ) );
  return center_dist;
}

inline void
update_manhatan_distance( const cv::Mat& data, const cv::Mat& centers, cv::Mat& distance )
{
  for( int i_cluster = 0; i_cluster < distance.cols; i_cluster++ )
  {
    for( int i_data = 0; i_data < distance.rows; i_data++ )
    {
      distance.at< float >( i_data, i_cluster ) = manhatan_distance( data, centers, i_data, i_cluster );
    }
  }
}

inline void
update_labels( const cv::Mat& dist, cv::Mat& _bestLabels )
{
  for( int i_data = 0; i_data < dist.rows; i_data++ )
  {
    int i_best = 0;
    double best_dist = std::numeric_limits< float >::max();
    for( int i_cluster = 0; i_cluster < dist.cols; i_cluster++ )
    {
      double cluster_dist = dist.at< float >( i_data, i_cluster );
      if( cluster_dist <= best_dist )
      {
        i_best = i_cluster;
        best_dist = cluster_dist;
      }
    }
    _bestLabels.at< int >( i_data, 0 ) = i_best;
  }
}


// copied from https://www.programmingalgorithms.com/algorithm/median/cpp/
float
median( std::vector< float > data )
{
  sort( data.begin(), data.end() );

  if( data.size() % 2 == 0 )
    return ( data[data.size() / 2 - 1] + data[data.size() / 2] ) / 2;
  else
    return data[data.size() / 2];
}

inline float
median( const cv::Mat& _data, const std::vector< int >& data_idx, int i_feature )
{
  if( data_idx.empty() )
    return 0;

  std::vector< float > values;
  for( auto i_data : data_idx )
  {
    values.emplace_back( _data.at< float >( i_data, i_feature ) );
  }

  return median( values );
}


inline void
update_medians( const cv::Mat& _data, const cv::Mat& _bestLabels, cv::Mat& _centers )
{
  std::vector< std::vector< int > > center_points;
  center_points.resize( _centers.rows );

  for( int i_data = 0; i_data < _bestLabels.rows; i_data++ )
  {
    center_points[_bestLabels.at< int >( i_data, 0 )].push_back( i_data );
  }

  for( int i_cluster = 0; i_cluster < _centers.rows; i_cluster++ )
  {
    for( int i_feature = 0; i_feature < _centers.cols; i_feature++ )
    {
      _centers.at< float >( i_cluster, i_feature ) = median( _data, center_points[i_cluster], i_feature );
    }
  }
}

inline std::set< int >
get_cluster_idx( const cv::Mat& labels )
{
  std::set< int > i_cluster;
  for( int i_row = 0; i_row < labels.rows; i_row++ )
  {
    i_cluster.insert( labels.at< int >( i_row, 0 ) );
  }
  return i_cluster;
}

inline bool
contains_empty_clusters( const cv::Mat& labels, int n_cluster )
{
  return get_cluster_idx( labels ).size() < n_cluster;
}

inline int
get_next_empty_cluster_id( const cv::Mat& labels, int n_cluster )
{
  auto cluster_idx = get_cluster_idx( labels );
  for( int i_cluster = 0; i_cluster < n_cluster; i_cluster++ )
    if( cluster_idx.find( i_cluster ) == std::end( cluster_idx ) )
      return i_cluster;
  return -1;
}

inline int
get_biggest_cluster_id( const cv::Mat& labels, int n_cluster )
{
  std::vector< int > count( n_cluster );
  for( int i_data = 0; i_data < labels.rows; i_data++ )
    count[labels.at< int >( i_data, 0 )] += 1;

  int i_max{}, max_val{};
  for( int i_cluster = 0; i_cluster < n_cluster; i_cluster++ )
  {
    if( count[i_cluster] > max_val )
    {
      i_max = i_cluster;
      max_val = count[i_cluster];
    }
  }
  return i_max;
}


template< typename F >
int
find_frame_id_from_center( const cv::Mat& data,
                           const cv::Mat& labels,
                           const cv::Mat& centers,
                           int i_cluster,
                           double init,
                           const F& pred )
{
  int i_selected{};
  double prev_dist{ init };

  for( int i_data = 0; i_data < data.rows; i_data++ )
  {
    if( labels.at< int >( i_data, 0 ) != i_cluster )
      continue;

    auto data_dist = manhatan_distance( data, centers, i_data, i_cluster );
    if( pred( data_dist, prev_dist ) )
    {
      i_selected = i_data;
      prev_dist = data_dist;
    }
  }
  return i_selected;
}


int
viame::find_closest_frame_id_to_center( const cv::Mat& data,
                                        const cv::Mat& labels,
                                        const cv::Mat& centers,
                                        int i_cluster )
{
  return find_frame_id_from_center( data, labels, centers, i_cluster, std::numeric_limits< double >::max(),
                                    []( double data_dist, double prev_dist ) { return data_dist < prev_dist; } );
}

int
viame::find_furthest_frame_id_from_center( const cv::Mat& data,
                                           const cv::Mat& labels,
                                           const cv::Mat& centers,
                                           int i_cluster )
{
  return find_frame_id_from_center( data, labels, centers, i_cluster, std::numeric_limits< double >::lowest(),
                                    []( double data_dist, double prev_dist ) { return data_dist > prev_dist; } );
}


inline void
make_sure_no_center_is_empty( const cv::Mat& data, cv::Mat& labels, cv::Mat& centers )
{
  auto n_cluster = centers.rows;
  // While empty cluster
  while( contains_empty_clusters( labels, n_cluster ) )
  {
    // Get id of empty
    auto empty_id = get_next_empty_cluster_id( labels, n_cluster );

    // Find biggest cluster
    auto biggest_id = get_biggest_cluster_id( labels, n_cluster );

    // Find farthest center point from center in cluster
    auto furthest_id = viame::find_furthest_frame_id_from_center( data, labels, centers, biggest_id );

    // Add point to empty cluster
    labels.at< int >( furthest_id, 0 ) = empty_id;

    // Update medians / labels
    update_medians( data, labels, centers );
  }
}

double
viame::kmedians( const cv::Mat& _data,
                 int K,
                 cv::Mat& _bestLabels,
                 cv::TermCriteria criteria,
                 int attempts,
                 int flags,
                 cv::Mat& _centers )
{
  // Implementation inspired by https://github.com/UBC-MDS/KMediansPy/blob/master/KMediansPy/KMedians.py
  // Init centers and labels using OCV Kmeans
  cv::kmeans( _data, K, _bestLabels, criteria, attempts, flags, _centers );

  cv::Mat prev_centers( _centers.rows, _centers.cols, CV_32FC1, cv::Scalar( 0 ) );
  cv::Mat dist( _data.rows, _centers.rows, CV_32FC1, cv::Scalar( 0 ) );
  do
  {
    prev_centers = _centers.clone();
    update_manhatan_distance( _data, _centers, dist );
    update_labels( dist, _bestLabels );
    update_medians( _data, _bestLabels, _centers );
    make_sure_no_center_is_empty( _data, _bestLabels, _centers );
  }
  while( are_centers_different( _centers, prev_centers ) );

  return cv::sum( dist )[0];
}