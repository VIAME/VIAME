/*ckwg +29
* Copyright 2016 by Kitware SAS, 2018-2019 Kitware, Inc.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*  * Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*
*  * Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
*  * Neither name of Kitware, Inc. nor the names of any contributors may be used
*    to endorse or promote products derived from this software without specific
*    prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef INTEGRATE_DEPTH_MAPS_CU_
#define INTEGRATE_DEPTH_MAPS_CU_

// STD include
#include <math.h>
#include <stdio.h>
#include <vector>
#include "cuda_error_check.h"

#define size4x4 16

//*****************************************************************************

// Define texture and constants
__constant__ double c_gridOrig[3];        // Origin of the output volume
__constant__ int3 c_gridDims;             // Dimensions of the output volume
__constant__ double c_gridSpacing[3];     // Spacing of the output volume
__constant__ int2 c_depthMapDims;         // Dimensions of all depths map
__constant__ double c_rayPotentialThick;  // Thickness threshold for the ray potential function
__constant__ double c_rayPotentialRho;    // Rho at the Y axis for the ray potential function
__constant__ double c_rayPotentialEta;
__constant__ double c_rayPotentialEpsilon;
__constant__ double c_rayPotentialDelta;
int grid_dims[3];


//*****************************************************************************
//     Truncated Signed Distance Function (TSDF) Parameter Description
//*****************************************************************************
//** Eta is a percentage of rho ( 0 < Eta < 1)
//** Epsilon is a percentage of rho ( 0 < Epsilon < 1)
//** Delta has to be superior to Thick
//
//                     'real distance' - 'depth value'
//                                     |
//                                     |
//                                     |         ---------------  Rho
//                                     |        /|             |
//                                     |       /               |
//                                     |      /  |             |
//                                     |     /                 |
//                                     |    /    |             |
//                                     |   /                   |
//                                     |  /      |             |
//                                     | /         Epsilon*Rho |______________
//                                     |/        |
//----------------------------------------------------------------------------
//                                    /
//                                   /
//                                  /
//--------------  Eta*rho          /
//             |                  /
//             |                 /
//             |                /
//             |               /
//             |              /
//             ---------------
//                            <--------->
//                               Thick
//             <----------------------->
//                        Delta
//*****************************************************************************

__device__ void computeVoxelCenter(int voxelCoordinate[3], double output[3])
{
  output[0] = c_gridOrig[0] + (voxelCoordinate[0] + 0.5) * c_gridSpacing[0];
  output[1] = c_gridOrig[1] + (voxelCoordinate[1] + 0.5) * c_gridSpacing[1];
  output[2] = c_gridOrig[2] + (voxelCoordinate[2] + 0.5) * c_gridSpacing[2];
}

//*****************************************************************************

//Apply a 3x4 matrix to a 3D points (assumes last row of M is 0, 0, 0, 1)
__device__ void transformFrom4Matrix(double M[size4x4], double point[3], double output[3])
{
  output[0] = M[0 * 4 + 0] * point[0] + M[0 * 4 + 1] * point[1] + M[0 * 4 + 2] * point[2] + M[0 * 4 + 3];
  output[1] = M[1 * 4 + 0] * point[0] + M[1 * 4 + 1] * point[1] + M[1 * 4 + 2] * point[2] + M[1 * 4 + 3];
  output[2] = M[2 * 4 + 0] * point[0] + M[2 * 4 + 1] * point[1] + M[2 * 4 + 2] * point[2] + M[2 * 4 + 3];
}

//*****************************************************************************

// Compute the norm of a 3 vec
__device__ double norm(double vec[3])
{
  return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

//*****************************************************************************

//Ray potential function which computes the increment to the current voxel
__device__ void rayPotential(double realDistance, double depthMapDistance, double& res)
{
  double diff = (realDistance - depthMapDistance);

  double absoluteDiff = abs(diff);
  // Can't divide by zero
  int sign = diff != 0 ? diff / absoluteDiff : 0;

  if (absoluteDiff > c_rayPotentialDelta)
    res = diff > 0 ? c_rayPotentialEpsilon * c_rayPotentialRho
                   : - c_rayPotentialEta * c_rayPotentialRho;
  else if (absoluteDiff > c_rayPotentialThick)
    res = c_rayPotentialRho * sign;
  else
    res = (c_rayPotentialRho / c_rayPotentialThick) * diff;
}

//*****************************************************************************

// Compute the voxel Id on a 1D table according to its 3D coordinates
__device__ int computeVoxelIDGrid(int coordinates[3])
{
  int dimX = c_gridDims.x;
  int dimY = c_gridDims.y;
  int i = coordinates[0];
  int j = coordinates[1];
  int k = coordinates[2];
  return (k*dimY + j)*dimX + i;
}

//*****************************************************************************

//Compute the pixel Id on a 1D table according to its 3D coordinates (third coordinate is not used)
__device__ int computeVoxelIDDepth(int coordinates[3])
{
  int dimX = c_depthMapDims.x;
  int dimY = c_depthMapDims.y;
  int x = coordinates[0];
  int y = coordinates[1];
  // /!\ vtkImageData has its origin at the bottom left, not top left
  return (dimX*(dimY - 1 - y)) + x;
}

//*****************************************************************************

// Main kernel for adding a depth map to the volume
__global__ void depthMapKernel(double* depths, double* weights, double matrixK[size4x4], double matrixRT[size4x4],
  double* output)
{
  // Get voxel coordinate according to thread id
  int voxelIndex[3] = { (int)threadIdx.x, (int)blockIdx.y, (int)blockIdx.z };

  double voxelCenterCoordinate[3];
  computeVoxelCenter(voxelIndex, voxelCenterCoordinate);

  // Transform voxel center from real coord to camera coords
  double voxelCenterCamera[3];
  transformFrom4Matrix(matrixRT, voxelCenterCoordinate, voxelCenterCamera);

  // Transform voxel center from camera coords to depth map homogeneous coords
  double voxelCenterHomogen[3];
  transformFrom4Matrix(matrixK, voxelCenterCamera, voxelCenterHomogen);
  if (voxelCenterHomogen[2] < 0)
    return;

  // Get voxel center on depth map coord
  double voxelCenterDepthMap[2];
  voxelCenterDepthMap[0] = voxelCenterHomogen[0] / voxelCenterHomogen[2];
  voxelCenterDepthMap[1] = voxelCenterHomogen[1] / voxelCenterHomogen[2];
  // Get real pixel position (approximation)
  int pixel[3];
  pixel[0] = round(voxelCenterDepthMap[0]);
  pixel[1] = round(voxelCenterDepthMap[1]);
  pixel[2] = 0;

  // Test if coordinate are inside depth map
  if (pixel[0] < 0 || pixel[1] < 0 || pixel[0] >= c_depthMapDims.x || pixel[1] >= c_depthMapDims.y)
    return;

  // Compute the ID on depthmap values according to pixel position and depth map dimensions
  int depthMapId = computeVoxelIDDepth(pixel);
  double depth = depths[depthMapId];
  double weight = weights ? weights[depthMapId] : 1.0;
  if (depth <= 0 || weight <= 0)
    return;

  int gridId = computeVoxelIDGrid(voxelIndex);  // Get the distance between voxel and camera
  double realDepth = voxelCenterCamera[2];
  double newValue;
  rayPotential(realDepth, depth, newValue);
  // Update the value to the output
  output[gridId] += weight * newValue;
}

//*****************************************************************************

// Initialize cuda constants
void cuda_initalize(int h_gridDims[3],     // Dimensions of the output volume
          double h_gridOrig[3],  // Origin of the output volume
          double h_gridSpacing[3], // Spacing of the output volume
          double h_rayPThick,
          double h_rayPRho,
          double h_rayPEta,
          double h_rayPEpsilon,
          double h_rayPDelta)
{
  CudaErrorCheck(cudaMemcpyToSymbol(c_gridDims, h_gridDims, 3 * sizeof(int)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_gridOrig, h_gridOrig, 3 * sizeof(double)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_gridSpacing, h_gridSpacing, 3 * sizeof(double)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_rayPotentialThick, &h_rayPThick, sizeof(double)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_rayPotentialRho, &h_rayPRho, sizeof(double)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_rayPotentialEta, &h_rayPEta, sizeof(double)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_rayPotentialEpsilon, &h_rayPEpsilon, sizeof(double)));
  CudaErrorCheck(cudaMemcpyToSymbol(c_rayPotentialDelta, &h_rayPDelta, sizeof(double)));

  grid_dims[0] = h_gridDims[0];
  grid_dims[1] = h_gridDims[1];
  grid_dims[2] = h_gridDims[2];
}

//*****************************************************************************

void launch_depth_kernel(double * d_depth, double * d_conf, int h_depthMapDims[2], double d_K[size4x4], double d_RT[size4x4], double* d_volume)
{
  // Organize threads into blocks and grids
  dim3 dimBlock(grid_dims[0], 1, 1); // nb threads on each block
  dim3 dimGrid(1, grid_dims[1], grid_dims[2]); // nb blocks on a grid
  CudaErrorCheck(cudaMemcpyToSymbol(c_depthMapDims, h_depthMapDims, 2 * sizeof(int)));
  CudaErrorCheck(cudaDeviceSynchronize());
  depthMapKernel << < dimGrid, dimBlock >> >(d_depth, d_conf, d_K, d_RT, d_volume);
  CudaErrorCheck(cudaPeekAtLastError());
  CudaErrorCheck(cudaDeviceSynchronize());
}


#endif
