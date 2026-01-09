/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_OPENCV_ELLIPSE_PROPOSAL_H
#define VIAME_OPENCV_ELLIPSE_PROPOSAL_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <limits>

namespace viame {
namespace opencv {

// Mathematical constants
constexpr float PI = 3.14159265f;
constexpr float INF = 1E+37f;

// Detection methods for prioritization
enum class DetectionMethod : unsigned int
{
  TEMPLATE  = 0,
  DOG       = 1,
  ADAPTIVE  = 2,
  CANNY     = 3,
  HISTOGRAM = 4,
  MULTIPLE1 = 5,
  MULTIPLE2 = 6,
  MULTIPLE3 = 7
};

// Feature size-related constants
constexpr unsigned int COLOR_BINS     = 32;
constexpr unsigned int COLOR_FEATURES = 122;
constexpr unsigned int GABOR_FEATURES = 36;
constexpr unsigned int SIZE_FEATURES  = 9;
constexpr unsigned int EDGE_FEATURES  = 137;
constexpr unsigned int HOG_FEATURES   = 1764;
constexpr unsigned int NUM_HOG        = 2;
constexpr unsigned int MAX_CLASSIFIERS = 30;

// Simple 2D point structure
struct Point2D
{
  Point2D() : r( 0 ), c( 0 ) {}
  Point2D( int _r, int _c ) : r( _r ), c( _c ) {}
  int r, c;
};

// Contour structure for edge-based features
struct Contour
{
  int label;                  // Contour Identifier
  float mag;                  // Contour Magnitude (Confidence)
  bool coversOct[8];          // Contour Octant Coverage around IP center
  std::vector< Point2D > pts; // Vector of points comprising Contour

  Contour() : label( 0 ), mag( 0.0f )
  {
    for( int i = 0; i < 8; i++ )
    {
      coversOct[i] = false;
    }
  }
};

// Object Proposal (Candidate Point) and associated information
//
// This struct stores location, stats for classification, features extracted
// around the candidate location, and preliminary classification results.
struct ObjectProposal
{
  // Ellipse location parameters
  double r;       // row position
  double c;       // column position
  double major;   // major axis
  double minor;   // minor axis
  double angle;   // rotation angle

  // Refined ellipse location (from edge feature cost function)
  double nr;
  double nc;
  double nmajor;
  double nminor;
  double nangle;

  // Detection method and magnitude
  DetectionMethod method;
  double magnitude;
  unsigned int methodRank;

  // Border flags
  bool isCorner;       // Is the candidate on an image boundary
  bool isSideBorder[8]; // Which octants are outside the image

  // Activity flag for processing
  bool isActive;

  // Features for classification (using modern containers)
  std::vector< double > colorFeatures;
  std::vector< double > gaborFeatures;
  std::vector< double > sizeFeatures;
  std::vector< cv::Mat > hogResults;
  double majorAxisMeters;

  // Used for color detectors
  cv::Mat summaryImage;
  cv::Mat colorQuadrants;
  int colorQR, colorQC;
  std::vector< int > colorBinCount;

  // Edge-based features
  bool hasEdgeFeatures;
  std::vector< double > edgeFeatures;

  // Color statistics
  float innerColorAvg[3];
  float outerColorAvg[3];
  std::shared_ptr< Contour > bestContour;
  std::shared_ptr< Contour > fullContour;

  // Classification results
  int designation;
  std::string filename;
  unsigned int classification;
  std::vector< double > classMagnitudes;

  // Default constructor
  ObjectProposal()
    : r( 0 ), c( 0 ), major( 0 ), minor( 0 ), angle( 0 ),
      nr( 0 ), nc( 0 ), nmajor( 0 ), nminor( 0 ), nangle( 0 ),
      method( DetectionMethod::DOG ), magnitude( 0 ), methodRank( 0 ),
      isCorner( false ), isActive( true ),
      majorAxisMeters( 0 ), colorQR( 0 ), colorQC( 0 ),
      hasEdgeFeatures( false ), designation( 0 ), classification( 0 )
  {
    colorFeatures.resize( COLOR_FEATURES, 0.0 );
    gaborFeatures.resize( GABOR_FEATURES, 0.0 );
    sizeFeatures.resize( SIZE_FEATURES, 0.0 );
    hogResults.resize( NUM_HOG );
    colorBinCount.resize( COLOR_BINS, 0 );
    edgeFeatures.resize( EDGE_FEATURES, 0.0 );
    classMagnitudes.resize( MAX_CLASSIFIERS, -std::numeric_limits< double >::max() );

    for( int i = 0; i < 3; i++ )
    {
      innerColorAvg[i] = 0.0f;
      outerColorAvg[i] = 0.0f;
    }

    for( int i = 0; i < 8; i++ )
    {
      isSideBorder[i] = false;
    }
  }
};

using ObjectProposalPtr = std::shared_ptr< ObjectProposal >;
using ObjectProposalVector = std::vector< ObjectProposalPtr >;

// Detection output structure (less information than proposal)
struct Detection
{
  std::string img;    // Image name

  // Object location
  double r;
  double c;
  double major;
  double minor;
  double angle;

  // Object contour (if exists)
  Contour cntr;

  // Possible object IDs and classification probabilities
  std::vector< std::string > classIDs;
  std::vector< double > classProbabilities;

  Detection()
    : r( 0 ), c( 0 ), major( 0 ), minor( 0 ), angle( 0 )
  {}
};

using DetectionPtr = std::shared_ptr< Detection >;
using DetectionVector = std::vector< DetectionPtr >;

// Utility functions for image access (modernized from old IplImage style)
inline float getPixel32f( const cv::Mat& img, int r, int c )
{
  return img.at< float >( r, c );
}

inline void setPixel32f( cv::Mat& img, int r, int c, float value )
{
  img.at< float >( r, c ) = value;
}

inline float getPixel32f( const cv::Mat& img, int r, int c, int chan )
{
  return img.at< cv::Vec3f >( r, c )[chan];
}

// Quick median calculation for image
inline float quickMedian( const cv::Mat& img, int maxToSample )
{
  std::vector< float > samples;
  samples.reserve( maxToSample );

  int step = std::max( 1, ( img.rows * img.cols ) / maxToSample );

  for( int i = 0; i < img.rows && static_cast<int>(samples.size()) < maxToSample; i++ )
  {
    for( int j = 0; j < img.cols && static_cast<int>(samples.size()) < maxToSample; j += step )
    {
      if( img.channels() == 1 )
      {
        samples.push_back( img.at< float >( i, j ) );
      }
      else
      {
        cv::Vec3f pixel = img.at< cv::Vec3f >( i, j );
        samples.push_back( ( pixel[0] + pixel[1] + pixel[2] ) / 3.0f );
      }
    }
  }

  if( samples.empty() )
  {
    return 0.0f;
  }

  std::nth_element( samples.begin(), samples.begin() + samples.size() / 2, samples.end() );
  return samples[samples.size() / 2];
}

// Round to nearest integer
inline int dround( double input )
{
  double frac = input - std::floor( input );
  if( frac < 0.5 )
  {
    return static_cast<int>( std::floor( input ) );
  }
  return static_cast<int>( std::ceil( input ) );
}

// Scale candidate position and size
inline void scaleProposal( ObjectProposal& cd, float sf )
{
  cd.r = cd.r * sf;
  cd.c = cd.c * sf;
  cd.major = cd.major * sf;
  cd.minor = cd.minor * sf;
}

// Deallocate proposals
inline void deallocateProposals( ObjectProposalVector& proposals )
{
  proposals.clear();
}

// Filter proposals by size range
inline void filterProposals( ObjectProposalVector& proposals,
                             float minSize, float maxSize,
                             bool dealloc = true )
{
  ObjectProposalVector filtered;

  for( auto& p : proposals )
  {
    if( p->major >= minSize && p->major <= maxSize )
    {
      filtered.push_back( p );
    }
  }

  if( dealloc )
  {
    proposals = std::move( filtered );
  }
}

} // namespace opencv
} // namespace viame

#endif /* VIAME_OPENCV_ELLIPSE_PROPOSAL_H */
