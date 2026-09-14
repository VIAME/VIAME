/* Mock implementation of StereoLibLX for Linux integration testing.
 *
 * This provides a functional stub of the SeaGIS StereoLibLX C++ API so
 * the VIAME seagis_measurement_process can be compiled, linked, and
 * exercised on Linux without the proprietary Windows library or a
 * license file.  All geometric results are approximate (simple pinhole
 * stereo model) but structurally correct so the pipeline runs end-to-end.
 */

#include <LX_StereoInterface.h>

#include <cmath>
#include <map>
#include <string>
#include <vector>

// ============================================================================
// C2DPt
// ============================================================================
C2DPt::C2DPt() : m_dx( 0.0 ), m_dy( 0.0 ) {}
C2DPt::C2DPt( const C2DPt& rhs ) : m_dx( rhs.m_dx ), m_dy( rhs.m_dy ) {}
C2DPt::C2DPt( double dx, double dy ) : m_dx( dx ), m_dy( dy ) {}
C2DPt::~C2DPt() {}

const double& C2DPt::X() const { return m_dx; }
const double& C2DPt::Y() const { return m_dy; }
double& C2DPt::X() { return m_dx; }
double& C2DPt::Y() { return m_dy; }

const C2DPt& C2DPt::operator=( const C2DPt& rhs ) { copy( rhs ); return *this; }
void C2DPt::copy( const C2DPt& rhs ) { m_dx = rhs.m_dx; m_dy = rhs.m_dy; }

// ============================================================================
// C3DPt
// ============================================================================
C3DPt::C3DPt() : m_dx( 0.0 ), m_dy( 0.0 ), m_dz( 0.0 ) {}
C3DPt::C3DPt( const C3DPt& rhs ) : m_dx( rhs.m_dx ), m_dy( rhs.m_dy ), m_dz( rhs.m_dz ) {}
C3DPt::C3DPt( double dx, double dy, double dz ) : m_dx( dx ), m_dy( dy ), m_dz( dz ) {}
C3DPt::~C3DPt() {}

const double& C3DPt::X() const { return m_dx; }
const double& C3DPt::Y() const { return m_dy; }
const double& C3DPt::Z() const { return m_dz; }
double& C3DPt::X() { return m_dx; }
double& C3DPt::Y() { return m_dy; }
double& C3DPt::Z() { return m_dz; }

const C3DPt& C3DPt::operator=( const C3DPt& rhs ) { copy( rhs ); return *this; }
void C3DPt::copy( const C3DPt& rhs ) { m_dx = rhs.m_dx; m_dy = rhs.m_dy; m_dz = rhs.m_dz; }

// ============================================================================
// CStereoImpl — internal state for the mock
// ============================================================================

struct camera_info
{
  std::string filename;
  std::string name;
  int rows = 1080;
  int cols = 1920;
  bool loaded = false;
};

struct camera_pair
{
  camera_info cameras[2]; // LEFT=0, RIGHT=1
};

class CStereoImpl
{
public:
  // Simple pinhole mock parameters
  static constexpr double FOCAL_PX   = 1500.0;  // pixels
  static constexpr double BASELINE   = 100.0;    // mm
  static constexpr double CX         = 960.0;    // principal point x
  static constexpr double CY         = 540.0;    // principal point y

  bool m_licence_ok = false;
  double m_image_sd = 1.0;

  std::map< unsigned int, camera_pair > m_pairs;

  // Epipolar line storage (per-pair)
  std::vector< C2DPt > m_epipolar_points;
};

// ============================================================================
// CStereoInt
// ============================================================================
CStereoInt::CStereoInt()
  : m_pImpl( new CStereoImpl )
{
}

CStereoInt::~CStereoInt()
{
  delete m_pImpl;
}

// ---------------------------------------------------------------------------
void CStereoInt::Version( int& nMajorVersion, int& nMinorVersion )
{
  nMajorVersion = 2;
  nMinorVersion = 10;
}

// ---------------------------------------------------------------------------
bool CStereoInt::SetLicenceKeys( const std::string& /*strKey1*/,
                                 const std::string& /*strKey2*/ )
{
  m_pImpl->m_licence_ok = true;
  return true;
}

bool CStereoInt::LicencePresent() const
{
  // Mock always reports licence as present
  return true;
}

// ---------------------------------------------------------------------------
std::string CStereoInt::ErrorString( RESULT res )
{
  switch( res )
  {
    case OK:                       return "OK";
    case INVALID_CAMERA_FILE:      return "Invalid camera file";
    case INVALID_LEFT_CAMERA:      return "Invalid left camera";
    case INVALID_RIGHT_CAMERA:     return "Invalid right camera";
    case INVALID_CAMERA:           return "Invalid camera";
    case DATA_RANGE_ERROR:         return "Data range error";
    case FAILED:                   return "Failed";
    case INVALID_LICENCE:          return "Invalid licence";
    case CAMERA_FILE_NOT_PERMITTED:return "Camera file not permitted";
    case INVALID_CAMERA_PAIR:      return "Invalid camera pair";
    default:                       return "Unknown error";
  }
}

// ---------------------------------------------------------------------------
RESULT CStereoInt::LoadCameraFile( unsigned int nPairID, CAMERA_ID camID,
                                   const std::string& strFileName )
{
  auto& pair = m_pImpl->m_pairs[nPairID];
  int idx = ( camID == LEFT ) ? 0 : 1;

  pair.cameras[idx].filename = strFileName;
  pair.cameras[idx].loaded = true;

  // Derive a simple name from the filename
  auto pos = strFileName.find_last_of( "/\\" );
  pair.cameras[idx].name = ( pos != std::string::npos )
    ? strFileName.substr( pos + 1 )
    : strFileName;

  return OK;
}

RESULT CStereoInt::GetCameraFile( unsigned int nPairID, CAMERA_ID camID,
                                  std::string& strFileName ) const
{
  auto it = m_pImpl->m_pairs.find( nPairID );
  if( it == m_pImpl->m_pairs.end() ) return INVALID_CAMERA_PAIR;

  int idx = ( camID == LEFT ) ? 0 : 1;
  if( !it->second.cameras[idx].loaded ) return INVALID_CAMERA;

  strFileName = it->second.cameras[idx].filename;
  return OK;
}

RESULT CStereoInt::GetCameraName( unsigned int nPairID, CAMERA_ID camID,
                                  std::string& strValue ) const
{
  auto it = m_pImpl->m_pairs.find( nPairID );
  if( it == m_pImpl->m_pairs.end() ) return INVALID_CAMERA_PAIR;

  int idx = ( camID == LEFT ) ? 0 : 1;
  if( !it->second.cameras[idx].loaded ) return INVALID_CAMERA;

  strValue = it->second.cameras[idx].name;
  return OK;
}

RESULT CStereoInt::GetCameraFormat( unsigned int nPairID, CAMERA_ID camID,
                                    int& nRows, int& nCols ) const
{
  auto it = m_pImpl->m_pairs.find( nPairID );
  if( it == m_pImpl->m_pairs.end() ) return INVALID_CAMERA_PAIR;

  int idx = ( camID == LEFT ) ? 0 : 1;
  if( !it->second.cameras[idx].loaded ) return INVALID_CAMERA;

  nRows = it->second.cameras[idx].rows;
  nCols = it->second.cameras[idx].cols;
  return OK;
}

unsigned int CStereoInt::GetCameraPairsUsed() const
{
  return static_cast< unsigned int >( m_pImpl->m_pairs.size() );
}

RESULT CStereoInt::RemoveCameraPair( unsigned int nPairID )
{
  auto it = m_pImpl->m_pairs.find( nPairID );
  if( it == m_pImpl->m_pairs.end() ) return INVALID_CAMERA_PAIR;

  m_pImpl->m_pairs.erase( it );
  return OK;
}

void CStereoInt::RemoveAllCameraPairs()
{
  m_pImpl->m_pairs.clear();
}

// ---------------------------------------------------------------------------
RESULT CStereoInt::GetUnits( unsigned int nPairID, std::string& strUnits ) const
{
  auto it = m_pImpl->m_pairs.find( nPairID );
  if( it == m_pImpl->m_pairs.end() ) return INVALID_CAMERA_PAIR;

  strUnits = "mm";
  return OK;
}

// ---------------------------------------------------------------------------
RESULT CStereoInt::SetImageMeasurementSD( double dValue )
{
  if( dValue <= 0.0 ) return DATA_RANGE_ERROR;
  m_pImpl->m_image_sd = dValue;
  return OK;
}

double CStereoInt::GetImageMeasurementSD( double* pdMin, double* pdMax ) const
{
  if( pdMin ) *pdMin = 0.1;
  if( pdMax ) *pdMax = 10.0;
  return m_pImpl->m_image_sd;
}

// ---------------------------------------------------------------------------
// Simple pinhole stereo intersection mock:
//   Z = focal * baseline / disparity
//   X = (u_left - cx) * Z / focal
//   Y = (v_avg  - cy) * Z / focal
// ---------------------------------------------------------------------------
RESULT CStereoInt::Intersect( unsigned int nPairID,
                              const C2DPt& ptLeft,
                              const C2DPt& ptRight,
                              C3DPt& ptAnswer,
                              double& dRMS,
                              C3DPt& ptSD ) const
{
  auto it = m_pImpl->m_pairs.find( nPairID );
  if( it == m_pImpl->m_pairs.end() ) return INVALID_CAMERA_PAIR;

  double disparity = ptLeft.X() - ptRight.X();
  if( std::abs( disparity ) < 0.01 )
  {
    disparity = ( disparity >= 0.0 ) ? 0.01 : -0.01;
  }

  double Z = CStereoImpl::FOCAL_PX * CStereoImpl::BASELINE / std::abs( disparity );
  double X = ( ptLeft.X() - CStereoImpl::CX ) * Z / CStereoImpl::FOCAL_PX;
  double v_avg = ( ptLeft.Y() + ptRight.Y() ) / 2.0;
  double Y = ( v_avg - CStereoImpl::CY ) * Z / CStereoImpl::FOCAL_PX;

  ptAnswer = C3DPt( X, Y, Z );

  // Approximate uncertainty (grows with depth squared / disparity)
  double sd_z = Z * Z / ( CStereoImpl::FOCAL_PX * CStereoImpl::BASELINE )
              * m_pImpl->m_image_sd;
  double sd_xy = Z / CStereoImpl::FOCAL_PX * m_pImpl->m_image_sd;

  ptSD = C3DPt( sd_xy, sd_xy, sd_z );
  dRMS = std::sqrt( sd_xy * sd_xy + sd_xy * sd_xy + sd_z * sd_z );

  return OK;
}

// ---------------------------------------------------------------------------
// Project a 3D point back to 2D (simple pinhole, left camera only for LEFT,
// shifted by baseline for RIGHT)
// ---------------------------------------------------------------------------
RESULT CStereoInt::ImageCoordinate( unsigned int nPairID, CAMERA_ID camID,
                                    const C3DPt& pt3D, C2DPt& ptImage ) const
{
  auto it = m_pImpl->m_pairs.find( nPairID );
  if( it == m_pImpl->m_pairs.end() ) return INVALID_CAMERA_PAIR;

  if( pt3D.Z() <= 0.0 ) return DATA_RANGE_ERROR;

  double x_offset = ( camID == RIGHT ) ? CStereoImpl::BASELINE : 0.0;

  double u = CStereoImpl::FOCAL_PX * ( pt3D.X() - x_offset ) / pt3D.Z()
           + CStereoImpl::CX;
  double v = CStereoImpl::FOCAL_PX * pt3D.Y() / pt3D.Z()
           + CStereoImpl::CY;

  ptImage = C2DPt( u, v );
  return OK;
}

// ---------------------------------------------------------------------------
double CStereoInt::Distance( const C3DPt& pt1, const C3DPt& pt1SD,
                             const C3DPt& pt2, const C3DPt& pt2SD,
                             double& dSD )
{
  double dx = pt2.X() - pt1.X();
  double dy = pt2.Y() - pt1.Y();
  double dz = pt2.Z() - pt1.Z();
  double dist = std::sqrt( dx * dx + dy * dy + dz * dz );

  // Propagate uncertainties (simplified)
  if( dist > 0.0 )
  {
    double sx = ( pt1SD.X() * pt1SD.X() + pt2SD.X() * pt2SD.X() ) * dx * dx;
    double sy = ( pt1SD.Y() * pt1SD.Y() + pt2SD.Y() * pt2SD.Y() ) * dy * dy;
    double sz = ( pt1SD.Z() * pt1SD.Z() + pt2SD.Z() * pt2SD.Z() ) * dz * dz;
    dSD = std::sqrt( sx + sy + sz ) / dist;
  }
  else
  {
    dSD = 0.0;
  }

  return dist;
}

// ---------------------------------------------------------------------------
int CStereoInt::MaxEpipolarPoints()
{
  return 200;
}

RESULT CStereoInt::EpipolarLine( unsigned int nPairID, CAMERA_ID camID,
                                 const C2DPt& ptImage,
                                 double dMinRange, double dMaxRange,
                                 int& nSz )
{
  auto it = m_pImpl->m_pairs.find( nPairID );
  if( it == m_pImpl->m_pairs.end() ) return INVALID_CAMERA_PAIR;

  if( dMinRange <= 0.0 || dMaxRange <= dMinRange )
  {
    nSz = 0;
    return DATA_RANGE_ERROR;
  }

  // Generate epipolar points in the OTHER camera.
  // For a rectified stereo pair the epipolar line is horizontal at the
  // same y coordinate, spanning disparity values between dMinRange and
  // dMaxRange (treated as depths).
  m_pImpl->m_epipolar_points.clear();

  int num_points = 100;
  m_pImpl->m_epipolar_points.reserve( num_points );

  for( int i = 0; i < num_points; ++i )
  {
    double t = static_cast< double >( i ) / ( num_points - 1 );
    double depth = dMinRange + t * ( dMaxRange - dMinRange );

    // Disparity at this depth
    double disp = CStereoImpl::FOCAL_PX * CStereoImpl::BASELINE / depth;

    double u;
    if( camID == LEFT )
    {
      // Point is in left camera → epipolar in right camera
      u = ptImage.X() - disp;
    }
    else
    {
      // Point is in right camera → epipolar in left camera
      u = ptImage.X() + disp;
    }

    m_pImpl->m_epipolar_points.push_back( C2DPt( u, ptImage.Y() ) );
  }

  nSz = static_cast< int >( m_pImpl->m_epipolar_points.size() );
  return OK;
}

RESULT CStereoInt::GetEpipolarPoint( unsigned int /*nPairID*/, int nIndex,
                                     C2DPt& Pt ) const
{
  if( nIndex < 0 ||
      nIndex >= static_cast< int >( m_pImpl->m_epipolar_points.size() ) )
  {
    return DATA_RANGE_ERROR;
  }

  Pt = m_pImpl->m_epipolar_points[nIndex];
  return OK;
}
