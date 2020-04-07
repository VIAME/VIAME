/*ckwg +29
 * Copyright 2012-2019 by Kitware, Inc.
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

 /**
 * \file
 * \brief Source file for cost_volume, computes costs in the world space
 */
#include "cost_volume.h"

#include <cstdio>
#include <fstream>

#include <vil/vil_bilin_interp.h>
#include <vil/algo/vil_sobel_3x3.h>
#include <vnl/vnl_double_3.h>
#include <vnl/vnl_double_2.h>
#include <vbl/vbl_array_2d.h>

#include <vil/vil_math.h>
#include <vil/vil_save.h>
#include <vil/vil_convert.h>

#include <limits>

#include <vital/logger/logger.h>

namespace kwiver {
namespace arrows {
namespace super3d {

bool
compute_world_cost_volume(const std::vector<vil_image_view<double> > &frames,
                          const std::vector<vpgl_perspective_camera<double> > &cameras,
                          world_space *ws,
                          unsigned int ref_frame,
                          unsigned int S,
                          vil_image_view<double> &cost_volume,
                          cost_volume_callback_t callback,
                          const std::vector<vil_image_view<bool> > &masks)
{
  static vital::logger_handle_t logger =
    vital::get_logger("arrows.super3d.compute_world_cost_volume");

  const vil_image_view<double> &ref = frames[ref_frame];
  cost_volume = vil_image_view<double>(ws->ni(), ws->nj(), 1, S);
  cost_volume.fill(0.0);

  std::vector<vpgl_perspective_camera<double> > warp_cams = ws->warp_cams(cameras, ref_frame);

  double s_step = 1.0/static_cast<double>(S);

  LOG_DEBUG(logger, "Computing cost volume of size ("
                     << cost_volume.ni() << ", "
                     << cost_volume.nj() << ", "
                     << cost_volume.nplanes() << ")");

  int ni = ws->ni(), nj = ws->nj();
  vil_image_view<double> warp_ref(ni, nj, 1), warp(ni, nj, 1);
  vil_image_view<bool> warp_ref_mask(ni, nj, 1), warp_mask(ni, nj, 1);

  vil_image_view<int> counts(ni, nj, 1);

  //Depths
  for (unsigned int k = 0; k < S; k++)
  {
    LOG_TRACE(logger, "Depth Layer: " << k);
    if (callback)
    {
      if (!callback(k))
      {
        return false;
      }
    }
    double s = (k + 0.5) * s_step;

    // Warp ref image to world volume
    // (does nothing if world space is aligned with ref camera)
    ws->warp_image_to_depth(ref, warp_ref, warp_cams[ref_frame],
                            s, ref_frame, -1.0);
    if (!masks.empty())
    {
      ws->warp_image_to_depth(masks[ref_frame], warp_ref_mask,
                              warp_cams[ref_frame], s, ref_frame, false);
    }

    counts.fill(0);

  //sum costs across all images into this depth slice
    for (unsigned int f = 0; f < frames.size(); f++)
    {
      if (f == ref_frame)
        continue;

      //Warp frame to world volume
      ws->warp_image_to_depth(frames[f], warp, warp_cams[f], s, f, -1.0);
      if (!masks.empty())
      {
        ws->warp_image_to_depth(masks[f], warp_mask, warp_cams[f], s, f, false);
      }

#pragma omp parallel for
      for (int64_t j = 0; j < warp_ref.nj(); j++)
      {
        for (unsigned int i = 0; i < warp_ref.ni(); i++)
        {
          if (warp(i,j) == -1 ||
              (!masks.empty() && (warp_ref_mask(i,j) || warp_mask(i,j))))
            continue;

          cost_volume(i, j, k) += fabs(warp_ref(i, j) - warp(i, j));
          counts(i,j) += 1;
        }
      }
    }

  //Normalize by counts
    #pragma omp parallel for
    for (int64_t j = 0; j < warp_ref.nj(); j++)
    {
      for (unsigned int i = 0; i < warp_ref.ni(); i++)
      {
        if (!masks.empty() && warp_ref_mask(i, j))
          continue;

        if (counts(i, j) == 0)
          cost_volume(i, j, k) = std::numeric_limits<double>::infinity();
        else
          cost_volume(i, j, k) /= (double)counts(i,j);
      }
    }
  }
  return true;
}

//*****************************************************************************

//Compute gradient weighting
void
compute_g(const vil_image_view<double> &ref_img,
  vil_image_view<double> &g,
  double alpha,
  double beta,
  vil_image_view<bool> *mask)
{
  g.set_size(ref_img.ni(), ref_img.nj(), 1);

  vil_image_view<double> ref_img_g;
  vil_sobel_3x3(ref_img, ref_img_g);

  bool invalid_mask = !mask ||
                      mask->ni() != ref_img.ni() ||
                      mask->nj() != ref_img.nj();

  for (unsigned int i = 0; i < ref_img_g.ni(); i++)
  {
    for (unsigned int j = 0; j < ref_img_g.nj(); j++)
    {
      if (invalid_mask || (*mask)(i, j))
      {
        double dx = ref_img_g(i, j, 0);
        double dy = ref_img_g(i, j, 1);
        double mag = sqrt(dx*dx + dy*dy);
        g(i, j) = exp(-alpha * mag);
      }
      else
        g(i, j) = 1.0;
    }
  }
}

//*****************************************************************************

void
save_cost_volume(const vil_image_view<double> &cost_volume,
                      const vil_image_view<double> &g_weight,
                      const char *file_name)
{
  static vital::logger_handle_t logger =
    vital::get_logger("arrows.super3d.save_cost_volume");
  LOG_DEBUG(logger, "Saving cost volume to " << file_name);

  FILE *file = std::fopen(file_name, "wb");

  unsigned int ni = cost_volume.ni(), nj = cost_volume.nj();
  unsigned int np = cost_volume.nplanes();
  fwrite(&ni, sizeof(unsigned int), 1, file);
  fwrite(&nj, sizeof(unsigned int), 1, file);
  fwrite(&np, sizeof(unsigned int), 1, file);

  for (unsigned int i = 0; i < ni; i++)
  {
    for (unsigned int j = 0; j < nj; j++)
    {
      for (unsigned int s = 0; s < np; s++)
      {
        fwrite(&cost_volume(i,j,s), sizeof(double), 1, file);
      }
    }
  }

  for (unsigned int i = 0; i < ni; i++)
    for (unsigned int j = 0; j < nj; j++)
      fwrite(&g_weight(i,j), sizeof(double), 1, file);

  fclose(file);
}

//*****************************************************************************

void
load_cost_volume(vil_image_view<double> &cost_volume,
                      vil_image_view<double> &g_weight,
                      const char *file_name)
{
  static vital::logger_handle_t logger =
    vital::get_logger("arrows.super3d.load_cost_volume");
  LOG_DEBUG(logger, "Loading cost volume from " << file_name);

  FILE *file = fopen(file_name, "rb");

  unsigned int ni, nj, np;
  if (fread(&ni, sizeof(unsigned int), 1, file) != 1 ||
      fread(&nj, sizeof(unsigned int), 1, file) != 1 ||
      fread(&np, sizeof(unsigned int), 1, file) != 1 )
  {
    LOG_ERROR(logger, "Error loading cost volume");
    return;
  }

  g_weight.set_size(ni, nj, 1);
  cost_volume = vil_image_view<double>(ni, nj, 1, np);

  for (unsigned int i = 0; i < ni; i++)
  {
    for (unsigned int j = 0; j < nj; j++)
    {
      for (unsigned int s = 0; s < np; s++)
      {
        if (fread(&cost_volume(i,j,s), sizeof(double), 1, file) != 1)
        {
          LOG_ERROR(logger, "Error loading cost volume");
          return;
        }
      }
    }
  }

  for (unsigned int i = 0; i < ni; i++)
    for (unsigned int j = 0; j < nj; j++)
      if (fread(&g_weight(i,j), sizeof(double), 1, file) != 1)
      {
        LOG_ERROR(logger, "Error loading cost volume");
        return;
      }

  fclose(file);
}

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver
