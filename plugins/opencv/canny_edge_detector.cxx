/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "canny_edge_detector.h"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <stack>

namespace viame {
namespace opencv {

//------------------------------------------------------------------------------
// GaussianEdges Implementation
//------------------------------------------------------------------------------

const float GaussianEdges::KERNEL_SIZE_PER_SIGMA = 6.0f;

cv::Mat GaussianEdges::gaussDerivVertical( const cv::Mat& input, double sigma )
{
  int filterSize = static_cast< int >( sigma * KERNEL_SIZE_PER_SIGMA );
  filterSize = filterSize + ( filterSize + 1 ) % 2;  // Make odd

  cv::Mat kernel( filterSize, 1, CV_32FC1 );
  int center = filterSize / 2;
  float sig2 = static_cast< float >( sigma * sigma );
  float sig3 = static_cast< float >( sigma * sig2 );

  for( int i = 0; i < filterSize; i++ )
  {
    float pos = static_cast< float >( i - center );
    float value = -( pos / sig3 ) * std::exp( -pos * pos / ( 2.0f * sig2 ) );
    kernel.at< float >( i, 0 ) = value;
  }

  cv::Mat output;
  cv::filter2D( input, output, CV_32F, kernel );
  return output;
}

cv::Mat GaussianEdges::gaussDerivHorizontal( const cv::Mat& input, double sigma )
{
  int filterSize = static_cast< int >( sigma * KERNEL_SIZE_PER_SIGMA );
  filterSize = filterSize + ( filterSize + 1 ) % 2;  // Make odd

  cv::Mat kernel( 1, filterSize, CV_32FC1 );
  int center = filterSize / 2;
  float sig2 = static_cast< float >( sigma * sigma );
  float sig3 = static_cast< float >( sigma * sig2 );

  for( int i = 0; i < filterSize; i++ )
  {
    float pos = static_cast< float >( i - center );
    float value = -( pos / sig3 ) * std::exp( -pos * pos / ( 2.0f * sig2 ) );
    kernel.at< float >( 0, i ) = value;
  }

  cv::Mat output;
  cv::filter2D( input, output, CV_32F, kernel );
  return output;
}

cv::Mat GaussianEdges::computeGradientMagnitude( const cv::Mat& input, double sigma )
{
  cv::Mat dx = gaussDerivHorizontal( input, sigma );
  cv::Mat dy = gaussDerivVertical( input, sigma );

  cv::Mat magnitude;
  cv::magnitude( dx, dy, magnitude );
  return magnitude;
}

cv::Mat GaussianEdges::computeGradientOrientation( const cv::Mat& input, double sigma )
{
  cv::Mat dx = gaussDerivHorizontal( input, sigma );
  cv::Mat dy = gaussDerivVertical( input, sigma );

  cv::Mat orientation;
  cv::phase( dx, dy, orientation, true );  // degrees
  return orientation;
}

void GaussianEdges::computeGradient( const cv::Mat& input, double sigma,
                                      cv::Mat& magnitude, cv::Mat& orientation )
{
  cv::Mat dx = gaussDerivHorizontal( input, sigma );
  cv::Mat dy = gaussDerivVertical( input, sigma );

  cv::magnitude( dx, dy, magnitude );
  cv::phase( dx, dy, orientation, true );  // degrees
}

//------------------------------------------------------------------------------
// CannyEdgeDetector Implementation
//------------------------------------------------------------------------------

CannyEdgeDetector::CannyEdgeDetector()
  : m_config()
{
}

CannyEdgeDetector::CannyEdgeDetector( const CannyDetectorConfig& config )
  : m_config( config )
{
}

CannyEdgeDetector::~CannyEdgeDetector()
{
}

void CannyEdgeDetector::setConfig( const CannyDetectorConfig& config )
{
  m_config = config;
}

CannyDetectorConfig CannyEdgeDetector::getConfig() const
{
  return m_config;
}

cv::Mat CannyEdgeDetector::getEdgeImage() const
{
  return m_lastEdgeImage.clone();
}

bool CannyEdgeDetector::circleFrom3Points( float r1, float c1,
                                            float r2, float c2,
                                            float r3, float c3,
                                            EdgeCircle& circle )
{
  // Calculate squared coordinates
  float sqr[6];
  sqr[0] = r1 * r1;
  sqr[1] = c1 * c1;
  sqr[2] = r2 * r2;
  sqr[3] = c2 * c2;
  sqr[4] = r3 * r3;
  sqr[5] = c3 * c3;

  float addA = sqr[0] + sqr[1];
  float addB = sqr[2] + sqr[3];
  float addC = sqr[4] + sqr[5];

  float lower1 = r1 * ( c3 - c2 ) + r2 * ( c1 - c3 ) + r3 * ( c2 - c1 );
  float lower2 = c1 * ( r3 - r2 ) + c2 * ( r1 - r3 ) + c3 * ( r2 - r1 );

  // Check for collinear points
  if( std::abs( lower1 ) < 1e-10 || std::abs( lower2 ) < 1e-10 )
  {
    return false;
  }

  circle.r = ( addA * ( c3 - c2 ) + addB * ( c1 - c3 ) + addC * ( c2 - c1 ) );
  circle.r = 0.5 * circle.r / lower1;

  circle.c = ( addA * ( r3 - r2 ) + addB * ( r1 - r3 ) + addC * ( r2 - r1 ) );
  circle.c = 0.5 * circle.c / lower2;

  circle.radius = std::sqrt( ( r1 - circle.r ) * ( r1 - circle.r ) +
                             ( c1 - circle.c ) * ( c1 - circle.c ) );

  return true;
}

void CannyEdgeDetector::calculateMSE( EdgeCircle& circle,
                                       const std::vector< cv::Point >& contour )
{
  double mse = 0.0;

  for( const auto& pt : contour )
  {
    double dr = ( pt.y - circle.r ) / circle.radius;
    double dc = ( pt.x - circle.c ) / circle.radius;
    double dist = std::sqrt( dr * dr + dc * dc );
    double error = dist - 1.0;
    mse += error * error;
  }

  circle.normalizedMSE = mse / contour.size();
}

void CannyEdgeDetector::calculateCoverage( EdgeCircle& circle,
                                            const std::vector< cv::Point >& contour )
{
  circle.contourSize = static_cast< int >( contour.size() );
  circle.coverage = static_cast< double >( circle.contourSize * circle.contourSize ) /
                    circle.radius;
}

void CannyEdgeDetector::linkEdgeContours(
  const cv::Mat& edges,
  std::vector< std::vector< cv::Point > >& contours )
{
  // Create working copy
  cv::Mat work = edges.clone();

  const uchar EDGE_MARKER = 255;
  const uchar USED_MARKER = 254;

  int height = work.rows;
  int width = work.cols;

  // 8-connected neighbor offsets
  const int dr[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
  const int dc[] = { -1, 0, 1, -1, 1, -1, 0, 1 };

  for( int r = 1; r < height - 1; r++ )
  {
    for( int c = 1; c < width - 1; c++ )
    {
      if( work.at< uchar >( r, c ) == EDGE_MARKER )
      {
        // Start new contour with flood-fill
        std::vector< cv::Point > contour;
        std::stack< cv::Point > stack;
        stack.push( cv::Point( c, r ) );

        while( !stack.empty() )
        {
          cv::Point pt = stack.top();
          stack.pop();

          // Bounds check
          if( pt.x < 1 || pt.y < 1 || pt.x >= width - 1 || pt.y >= height - 1 )
          {
            continue;
          }

          // Check if already processed
          if( work.at< uchar >( pt.y, pt.x ) != EDGE_MARKER )
          {
            continue;
          }

          // Mark as used and add to contour
          work.at< uchar >( pt.y, pt.x ) = USED_MARKER;
          contour.push_back( pt );

          // Add 8-connected neighbors
          for( int i = 0; i < 8; i++ )
          {
            stack.push( cv::Point( pt.x + dc[i], pt.y + dr[i] ) );
          }
        }

        if( !contour.empty() )
        {
          contours.push_back( std::move( contour ) );
        }
      }
    }
  }
}

bool CannyEdgeDetector::detect( const cv::Mat& grayImage,
                                 ObjectProposalVector& proposals,
                                 float minRadius, float maxRadius,
                                 float scale )
{
  if( grayImage.empty() )
  {
    return false;
  }

  // Convert to 8-bit if needed
  cv::Mat gray8u;
  if( grayImage.depth() == CV_32F )
  {
    grayImage.convertTo( gray8u, CV_8U, 255.0 );
  }
  else if( grayImage.depth() == CV_8U )
  {
    gray8u = grayImage.clone();
  }
  else
  {
    grayImage.convertTo( gray8u, CV_8U );
  }

  // Apply blur
  cv::Mat blurred;
  cv::blur( gray8u, blurred, cv::Size( m_config.blurKernelSize, m_config.blurKernelSize ) );

  // Run Canny edge detection
  cv::Canny( blurred, m_lastEdgeImage,
             m_config.cannyLowThreshold,
             m_config.cannyHighThreshold,
             m_config.cannyApertureSize );

  // Link edge pixels into contours
  std::vector< std::vector< cv::Point > > contours;
  linkEdgeContours( m_lastEdgeImage, contours );

  // Fit circles to contours
  std::vector< EdgeCircle > circles;

  for( const auto& contour : contours )
  {
    int csize = static_cast< int >( contour.size() );

    // Skip small contours
    if( csize < m_config.minContourSize )
    {
      continue;
    }

    // For small contours, use single 3-point fit
    if( csize < 16 )
    {
      int i1 = 1;
      int i2 = csize / 2;
      int i3 = csize - 2;

      EdgeCircle circle;
      bool valid = circleFrom3Points(
        static_cast< float >( contour[i1].y ), static_cast< float >( contour[i1].x ),
        static_cast< float >( contour[i2].y ), static_cast< float >( contour[i2].x ),
        static_cast< float >( contour[i3].y ), static_cast< float >( contour[i3].x ),
        circle );

      if( valid && circle.radius > minRadius * 1.5 && circle.radius < maxRadius )
      {
        calculateMSE( circle, contour );
        calculateCoverage( circle, contour );
        circles.push_back( circle );
      }
    }
    else
    {
      // For larger contours, try multiple 3-point combinations
      int i1 = 0;
      int i2 = static_cast< int >( 0.2 * csize );
      int i3 = static_cast< int >( 0.4 * csize );
      int i4 = static_cast< int >( 0.6 * csize );
      int i5 = static_cast< int >( 0.8 * csize );
      int i6 = csize - 1;

      // Try three different combinations
      std::vector< std::tuple< int, int, int > > combos = {
        { i1, i4, i6 }, { i2, i4, i6 }, { i1, i3, i5 }
      };

      for( const auto& combo : combos )
      {
        int a = std::get< 0 >( combo );
        int b = std::get< 1 >( combo );
        int c = std::get< 2 >( combo );

        EdgeCircle circle;
        bool valid = circleFrom3Points(
          static_cast< float >( contour[a].y ), static_cast< float >( contour[a].x ),
          static_cast< float >( contour[b].y ), static_cast< float >( contour[b].x ),
          static_cast< float >( contour[c].y ), static_cast< float >( contour[c].x ),
          circle );

        if( valid && circle.radius > minRadius && circle.radius < maxRadius )
        {
          calculateMSE( circle, contour );
          calculateCoverage( circle, contour );
          circles.push_back( circle );
        }
      }
    }
  }

  // Sort by coverage/MSE ratio (higher is better)
  std::sort( circles.begin(), circles.end(),
    []( const EdgeCircle& a, const EdgeCircle& b )
    {
      double scoreA = ( a.normalizedMSE > 0 ) ? a.coverage / a.normalizedMSE : 0;
      double scoreB = ( b.normalizedMSE > 0 ) ? b.coverage / b.normalizedMSE : 0;
      return scoreA > scoreB;
    } );

  // Create proposals from top circles
  int numProposals = std::min( static_cast< int >( circles.size() ), m_config.maxProposals );

  for( int i = 0; i < numProposals; i++ )
  {
    const EdgeCircle& circle = circles[i];

    auto proposal = std::make_shared< ObjectProposal >();
    proposal->r = circle.r * scale;
    proposal->c = circle.c * scale;
    proposal->major = circle.radius * scale;
    proposal->minor = circle.radius * scale;
    proposal->angle = 0.0;
    proposal->method = DetectionMethod::CANNY;
    proposal->magnitude = circle.normalizedMSE * circle.contourSize;

    proposals.push_back( proposal );
  }

  return true;
}

} // namespace opencv
} // namespace viame
