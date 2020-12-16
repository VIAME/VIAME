// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "vtkKwiverCamera.h"

#include <vital/io/camera_io.h>
#include <vital/types/vector.h>

#include <vtkMath.h>
#include <vtkMatrix4x4.h>
#include <vtkObjectFactory.h>

namespace // anonymous
{

//-----------------------------------------------------------------------------
  void BuildCamera(kwiver::arrows::vtk::vtkKwiverCamera* out,
                   kwiver::vital::camera_perspective_sptr const& in,
                   kwiver::vital::camera_intrinsics_sptr const& ci)
{
  // Get camera parameters
  auto const pixelAspect = ci->aspect_ratio();
  auto const focalLength = ci->focal_length();

  int imageWidth, imageHeight;
  out->GetImageDimensions(imageWidth, imageHeight);

  double const aspectRatio = pixelAspect * imageWidth / imageHeight;
  out->SetAspectRatio(aspectRatio);

  double const fov =
    vtkMath::DegreesFromRadians(2.0 * atan(0.5 * imageHeight / focalLength));
  out->SetViewAngle(fov);

  // Compute camera vectors from matrix
  auto const& rotationMatrix = in->rotation().quaternion().toRotationMatrix();

  auto const up = -rotationMatrix.row(1).transpose();
  auto const view = rotationMatrix.row(2).transpose();
  auto const center = in->center();

  out->SetPosition(center[0], center[1], center[2]);
  out->SetViewUp(up[0], up[1], up[2]);

  auto const& focus = center + (view * out->GetDistance() / view.norm());
  out->SetFocalPoint(focus[0], focus[1], focus[2]);
}

} // namespace <anonymous>

namespace kwiver {
namespace arrows {
namespace vtk {

vtkStandardNewMacro(vtkKwiverCamera);

//-----------------------------------------------------------------------------
vtkKwiverCamera::vtkKwiverCamera()
{
  this->ImageDimensions[0] = this->ImageDimensions[1] = -1;
  this->AspectRatio = 1;
}

//-----------------------------------------------------------------------------
kwiver::vital::camera_perspective_sptr vtkKwiverCamera::GetCamera() const
{
  return this->KwiverCamera;
}

//-----------------------------------------------------------------------------
void vtkKwiverCamera::SetCamera(kwiver::vital::camera_perspective_sptr const& camera)
{
  this->KwiverCamera = camera;
}

//-----------------------------------------------------------------------------
bool vtkKwiverCamera::ProjectPoint(kwiver::vital::vector_3d const& in,
                                  double (&out)[2])
{
  if (this->KwiverCamera->depth(in) < 0.0)
  {
    // if the projection is invalid, move the point to infinity
    // so that it doesn't render in the camera view
    out[0] = std::numeric_limits<double>::infinity();
    out[1] = std::numeric_limits<double>::infinity();
    return false;
  }

  auto const& ppos = this->KwiverCamera->project(in);
  // Ignore points that are very far from the image.
  // Including points that are too far away can degrade the precision
  // of location of points in the image.
  int const& w_max = 10 * this->ImageDimensions[0];
  int const& h_max = 10 * this->ImageDimensions[1];
  if (ppos[0] < -w_max || ppos[0] > w_max ||
    ppos[1] < -h_max || ppos[1] > h_max)
  {
    return false;
  }

  out[0] = ppos[0];
  out[1] = ppos[1];
  return true;
}
/**
  *
  * WARNING: The convention here is that depth is NOT the distance between the
  * camera center and the 3D point but the distance between the projection of
  * the 3D point on the optical axis and the optical center.
*/

//-----------------------------------------------------------------------------
kwiver::vital::vector_3d vtkKwiverCamera::UnprojectPoint(
  double pixel[2], double depth)
{
  // Build camera matrix
  auto const T = this->KwiverCamera->translation();
  auto const R = this->KwiverCamera->rotation().matrix();

  auto const inPoint = kwiver::vital::vector_2d{pixel[0], pixel[1]};
  auto const normPoint = this->KwiverCamera->intrinsics()->unmap(inPoint);

  auto const homogenousPoint = kwiver::vital::vector_3d{normPoint[0] * depth,
                                                        normPoint[1] * depth,
                                                        depth};

  return kwiver::vital::vector_3d(R.transpose() * (homogenousPoint - T));
}

//-----------------------------------------------------------------------------
kwiver::vital::vector_3d vtkKwiverCamera::UnprojectPoint(double pixel[2])
{
  auto const depth = this->KwiverCamera->depth(kwiver::vital::vector_3d(0, 0, 0));
  return this->UnprojectPoint(pixel, depth);
}

//-----------------------------------------------------------------------------
double vtkKwiverCamera::Depth(kwiver::vital::vector_3d const& point) const
{
  return this->KwiverCamera->depth(point);
}

//-----------------------------------------------------------------------------
void vtkKwiverCamera::ScaleK(double factor)
{
  auto K = this->KwiverCamera->intrinsics()->as_matrix();

  K(0, 0) *= factor;
  K(0, 1) *= factor;
  K(0, 2) *= factor;
  K(1, 1) *= factor;
  K(1, 2) *= factor;

  kwiver::vital::simple_camera_intrinsics newIntrinsics(K);

  kwiver::vital::simple_camera_perspective scaledCamera(this->KwiverCamera->center(),
                                                        this->KwiverCamera->rotation(),
                                                        newIntrinsics);
  auto cam_ptr =
    std::dynamic_pointer_cast<kwiver::vital::camera_perspective>(scaledCamera.clone());
  SetCamera(cam_ptr);
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkKwiverCamera> vtkKwiverCamera::ScaledK(double factor)
{
  auto newCam = vtkSmartPointer<vtkKwiverCamera>::New();
  newCam->DeepCopy(this);

  newCam->ScaleK(factor);

  return newCam;
}

//-----------------------------------------------------------------------------
vtkSmartPointer<vtkKwiverCamera> vtkKwiverCamera::CropCamera(int i0, int ni, int j0, int nj)
{
  kwiver::vital::simple_camera_intrinsics newIntrinsics(*this->KwiverCamera->intrinsics());
  kwiver::vital::vector_2d pp = newIntrinsics.principal_point();

  pp[0] -= i0;
  pp[1] -= j0;

  newIntrinsics.set_principal_point(pp);

  kwiver::vital::simple_camera_perspective cropCam(this->KwiverCamera->center(),
                                                   this->KwiverCamera->rotation(),
                                                   newIntrinsics);

  auto cam_ptr =
    std::dynamic_pointer_cast<kwiver::vital::camera_perspective>(cropCam.clone());

  auto newCam = vtkSmartPointer<vtkKwiverCamera>::New();
  newCam->DeepCopy(this);
  newCam->SetCamera(cam_ptr);
  newCam->SetImageDimensions(ni, nj);

  return newCam;
}

//-----------------------------------------------------------------------------
bool vtkKwiverCamera::Update()
{
  auto const& ci = this->KwiverCamera->intrinsics();

  if (this->ImageDimensions[0] == -1 || this->ImageDimensions[1] == -1)
  {
    // Guess image size
    const kwiver::vital::vector_2d s = ci->principal_point() * 2.0;
    this->ImageDimensions[0] = s[0];
    this->ImageDimensions[1] = s[1];
  }

  BuildCamera(this, this->KwiverCamera, ci);

  // here for now, but this is something we actually want to be a property
  // of the representation... that is, the depth (size) displayed for the camera
  // as determined by the far clipping plane
  auto const depth = 15.0;
  this->SetClippingRange(0.01, depth);
  return true;
}

//-----------------------------------------------------------------------------
void vtkKwiverCamera::GetFrustumPlanes(double planes[24])
{
  // Need to add timing (modfied time) logic to determine if need to Update()
  this->Superclass::GetFrustumPlanes(this->AspectRatio, planes);
}

//-----------------------------------------------------------------------------
void vtkKwiverCamera::GetTransform(vtkMatrix4x4* out, double const plane[4])
{
  // Build camera matrix
  auto const k = this->KwiverCamera->intrinsics()->as_matrix();
  auto const t = this->KwiverCamera->translation();
  auto const r = this->KwiverCamera->rotation().matrix();

  auto const kr = kwiver::vital::matrix_3x3d(k * r);
  auto const kt = kwiver::vital::vector_3d(k * t);

  out->SetElement(0, 0, kr(0, 0));
  out->SetElement(0, 1, kr(0, 1));
  out->SetElement(0, 3, kr(0, 2));
  out->SetElement(1, 0, kr(1, 0));
  out->SetElement(1, 1, kr(1, 1));
  out->SetElement(1, 3, kr(1, 2));
  out->SetElement(3, 0, kr(2, 0));
  out->SetElement(3, 1, kr(2, 1));
  out->SetElement(3, 3, kr(2, 2));

  out->SetElement(0, 3, kt[0]);
  out->SetElement(1, 3, kt[1]);
  out->SetElement(3, 3, kt[2]);

  // Insert plane coefficients into matrix to build plane-to-image projection
  out->SetElement(2, 0, plane[0]);
  out->SetElement(2, 1, plane[1]);
  out->SetElement(2, 2, plane[2]);
  out->SetElement(2, 3, plane[3]);

  // Invert to get image-to-plane projection
  out->Invert();
}

//-----------------------------------------------------------------------------
void vtkKwiverCamera::DeepCopy(vtkKwiverCamera* source)
{
  this->Superclass::DeepCopy(source);

  this->ImageDimensions[0] = source->ImageDimensions[0];
  this->ImageDimensions[1] = source->ImageDimensions[1];
  this->AspectRatio = source->AspectRatio;
  this->KwiverCamera = source->KwiverCamera;
}

//-----------------------------------------------------------------------------
void vtkKwiverCamera::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

} //end namespace vtk
} //end namespace arrows
} //end namespace kwiver
