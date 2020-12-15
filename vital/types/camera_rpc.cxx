// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of \link kwiver::vital::camera_rpc
 * camera_rpc \endlink class
 */

#include <vital/types/camera_rpc.h>
#include <vital/io/eigen_io.h>
#include <Eigen/Geometry>

#include <iomanip>

namespace kwiver {
namespace vital {

camera_rpc
::camera_rpc()
  : m_logger( kwiver::vital::get_logger( "vital.camera_rpc" ) )
{
}

/// Project a 3D point into a 2D image point
vector_2d
camera_rpc
::project( const vector_3d& pt ) const
{
  // Normalize points
  vector_3d norm_pt = ( pt - world_offset() ).cwiseQuotient( world_scale() );

  // Calculate polynomials
  vector_4d polys = this->rpc_coeffs()*this->power_vector(norm_pt);
  vector_2d image_pt( polys[0] / polys[1], polys[2] / polys[3]);

  // Un-normalize
  return image_pt.cwiseProduct( image_scale() ) + image_offset();
}

/// Project a 2D image point to a 3D point in space
vector_3d
camera_rpc
::back_project( const vector_2d& image_pt, double elev ) const
{
  // Normalize image point
  vector_2d norm_pt =
    ( image_pt - image_offset() ).cwiseQuotient( image_scale() );
  auto norm_elev = ( elev - world_offset()[2] ) / world_scale()[2];

  // Use a first order approximation to the RPC to initialize.
  // This sets all non-linear terms of the RPC to zero and then forms
  // a least squares solution to invert the mapping.
  matrix_2x2d A;
  vector_3d rslt( 0., 0., norm_elev );
  vector_2d b;

  A.block<1, 2>( 0, 0 ) = rpc_coeffs().block<1, 2>( 0, 1 )
                           - norm_pt[0] * rpc_coeffs().block<1, 2>( 1, 1 );
  A.block<1, 2>( 1, 0 ) = rpc_coeffs().block<1, 2>( 2, 1 );
                           - norm_pt[1] * rpc_coeffs().block<1, 2>( 3, 1 );

  b[0] = ( rpc_coeffs()( 1, 0 ) + norm_elev*rpc_coeffs()( 1, 3 ) )*norm_pt[0]
         - ( rpc_coeffs()( 0, 0 ) + norm_elev*rpc_coeffs()( 0, 3 ) );
  b[1] = ( rpc_coeffs()( 3, 0 ) + norm_elev*rpc_coeffs()( 3, 3 ) )*norm_pt[1]
         - ( rpc_coeffs()( 2, 0 ) + norm_elev*rpc_coeffs()( 2, 3 ) );

  rslt.head(2) = A.colPivHouseholderQr().solve( b );

  // Apply gradient descendent until convergence. Should converge
  // in a few interations.
  for ( int i = 0; i < 10; ++i )
  {
    matrix_2x2d J;
    vector_2d pt;

    this->jacobian( rslt, J, pt );

    vector_2d step = J.colPivHouseholderQr().solve( norm_pt - pt );
    rslt.head( 2 ) += step;
    if ( step.cwiseAbs().maxCoeff() < 1.e-16 )
    {
      break;
    }
  }

  return rslt.cwiseProduct( this->world_scale() ) + this->world_offset();
}

Eigen::Matrix<double, 20, 1>
camera_rpc
::power_vector( const vector_3d& pt )
{
  // Form the monomials in homogeneous form
  double w  = 1.0;
  double x = pt.x();
  double y = pt.y();
  double z = pt.z();

  double xx = x * x;
  double xy = x * y;
  double xz = x * z;
  double yy = y * y;
  double yz = y * z;
  double zz = z * z;
  double xxx = xx * x;
  double xxy = xx * y;
  double xxz = xx * z;
  double xyy = xy * y;
  double xyz = xy * z;
  double xzz = xz * z;
  double yyy = yy * y;
  double yyz = yy * z;
  double yzz = yz * z;
  double zzz = zz * z;

  // Fill in vector
  Eigen::Matrix<double, 20, 1> retVec;
  retVec << w, x, y, z, xy, xz, yz, xx, yy, zz,
            xyz, xxx, xyy, xzz, xxy, yyy, yzz, xxz, yyz, zzz;
  return retVec;
}

void
simple_camera_rpc
::update_partial_deriv() const
{
  std::vector<int> dx_ind = { 1, 7, 4, 5, 14 ,17 ,10, 11, 12, 13 };
  std::vector<int> dy_ind = { 2, 4, 8, 6, 12, 10, 18, 14, 15, 16 };
  for ( int i = 0; i < 10; ++i )
  {
    double pwr = 1.0;
    if ( i == 1 || i == 4 || i == 5 )
    {
      pwr = 2.;
    }
    else if ( i == 7 )
    {
      pwr = 3.;
    }
    dx_coeffs_.block<4, 1>( 0, i ) =
      pwr * rpc_coeffs().block<4, 1>( 0, dx_ind[i] );
    pwr = 1.0;
    if ( i == 2 || i == 4 || i == 6 )
    {
      pwr = 2.;
    }
    else if ( i == 8 )
    {
      pwr = 3.;
    }
    dy_coeffs_.block<4, 1>( 0, i ) =
      pwr * rpc_coeffs().block<4, 1>( 0, dy_ind[i] );
  }
}

void
simple_camera_rpc
::jacobian( const vector_3d& pt, matrix_2x2d& J, vector_2d& norm_pt ) const
{
  Eigen::Matrix<double, 20, 1> pv = this->power_vector( pt );
  vector_4d ply = this->rpc_coeffs() * pv;
  vector_4d dx_ply = this->dx_coeffs_ * pv.head(10);
  vector_4d dy_ply = this->dy_coeffs_ * pv.head(10);

  J( 0, 0 ) = ( ply[1] * dx_ply[0] - ply[0] * dx_ply[1] ) / ( ply[1] * ply[1] );
  J( 0, 1 ) = ( ply[1] * dy_ply[0] - ply[0] * dy_ply[1] ) / ( ply[1] * ply[1] );
  J( 1, 0 ) = ( ply[3] * dx_ply[2] - ply[2] * dx_ply[3] ) / ( ply[3] * ply[3] );
  J( 1, 1 ) = ( ply[3] * dy_ply[2] - ply[2] * dy_ply[3] ) / ( ply[3] * ply[3] );

  norm_pt << ply[0] / ply[1], ply[2] / ply[3];
}

} } // end namespace
