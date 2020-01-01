#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                         const int height, const int width,
                                         scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;
  // do bilinear interpolation
  scalar_t lt = bottom_data[y_low * width + x_low];
  scalar_t rt = bottom_data[y_low * width + x_high];
  scalar_t lb = bottom_data[y_high * width + x_low];
  scalar_t rb = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

  return val;
}

template <typename scalar_t>
__global__ void ROIAlignForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the aligned output
    int n = index;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    // scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    // scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);

    // scalar_t bin_size_h = roi_height / pooled_height;
    // scalar_t bin_size_w = roi_width / pooled_width;
    // original
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w + 1., 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h + 1., 0.);
    scalar_t bin_size_h = roi_height / (pooled_height - 1.);
    scalar_t bin_size_w = roi_width / (pooled_width - 1.);

    // const scalar_t *offset_bottom_data =
    //     bottom_data + (roi_batch_ind * channels + c) * height * width;

    // int sample_num_h = (sample_num > 0)
    //                        ? sample_num
    //                        : ceil(roi_height / pooled_height);  // e.g., = 2
    // int sample_num_w =
    //     (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    scalar_t h = (scalar_t)ph * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)pw * bin_size_w + roi_start_w;

    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);

    int img_start = roi_batch_ind * channels * height * width;

    if (h < 0 || h >= height || w < 0 || w >= width)
    {
        top_data[index] = 0.;
    }else{
        scalar_t h_ratio = h - (scalar_t)hstart;
        scalar_t w_ratio = w - (scalar_t)wstart;
        int upleft = img_start + (c * height + hstart) * width + wstart;
        int upright = upleft + 1;
        int downleft = upleft + width;
        int downright = downleft + 1;
        top_data[index] = bottom_data[upleft] * (1. - h_ratio) * (1. - w_ratio)
                    + bottom_data[upright] * (1. - h_ratio) * w_ratio
                    + bottom_data[downleft] * h_ratio * (1. - w_ratio)
                    + bottom_data[downright] * h_ratio * w_ratio;
    }
  }
}

int ROIAlignForwardLaucher(const at::Tensor features, const at::Tensor rois,
                           const float spatial_scale, const int sample_num,
                           const int channels, const int height,
                           const int width, const int num_rois,
                           const int pooled_height, const int pooled_width,
                           at::Tensor output) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "ROIAlignLaucherForward", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();

        ROIAlignForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sample_num, channels, height, width, pooled_height,
                pooled_width, top_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}

template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(const int height, const int width,
                                              scalar_t y, scalar_t x,
                                              scalar_t &w1, scalar_t &w2,
                                              scalar_t &w3, scalar_t &w4,
                                              int &x_low, int &x_high,
                                              int &y_low, int &y_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename scalar_t>
__global__ void ROIAlignBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sample_num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the aligned output
    int n = index;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w + 1., 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h + 1., 0.);

    scalar_t bin_size_h = roi_height / (pooled_height - 1.);
    scalar_t bin_size_w = roi_width / (pooled_width - 1.);

    // scalar_t *offset_bottom_diff =
    //     bottom_diff + (roi_batch_ind * channels + c) * height * width;
    // int offset_top = (n * channels + c) * pooled_height * pooled_width +
    //                  ph * pooled_width + pw;
    // scalar_t offset_top_diff = top_diff[offset_top];

    // int sample_num_h = (sample_num > 0)
    //                        ? sample_num
    //                        : ceil(roi_height / pooled_height);  // e.g., = 2
    // int sample_num_w =
    //     (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    // const scalar_t count = (scalar_t)(sample_num_h * sample_num_w);

    scalar_t h = (scalar_t)ph * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)pw * bin_size_w + roi_start_w;

    int hstart = fminf(floor(h), height - 2);
    int wstart = fminf(floor(w), width - 2);
    
    int img_start = roi_batch_ind * channels * height * width;

    if (!(h < 0 || h >= height || w < 0 || w >= width))
    {
        scalar_t h_ratio = h - (scalar_t)(hstart);
        scalar_t w_ratio = w - (scalar_t)(wstart);
        int upleft = img_start + (c * height + hstart) * width + wstart;
        int upright = upleft + 1;
        int downleft = upleft + width;
        int downright = downleft + 1;

        atomicAdd(bottom_diff + upleft, top_diff[index] * (1. - h_ratio) * (1 - w_ratio));
        atomicAdd(bottom_diff + upright, top_diff[index] * (1. - h_ratio) * w_ratio);
        atomicAdd(bottom_diff + downleft, top_diff[index] * h_ratio * (1 - w_ratio));
        atomicAdd(bottom_diff + downright, top_diff[index] * h_ratio * w_ratio);
    }
  }
}

int ROIAlignBackwardLaucher(const at::Tensor top_grad, const at::Tensor rois,
                            const float spatial_scale, const int sample_num,
                            const int channels, const int height,
                            const int width, const int num_rois,
                            const int pooled_height, const int pooled_width,
                            at::Tensor bottom_grad) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "ROIAlignLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        ROIAlignBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data, spatial_scale, sample_num,
                channels, height, width, pooled_height, pooled_width,
                bottom_diff);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}

template <typename scalar_t>
__global__ void ROIAlignAdaForward(const int nthreads, const scalar_t *bottom_data,
                                const scalar_t *bottom_rois,
                                const scalar_t spatial_scale,
                                const int sample_num, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                scalar_t *top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the aligned output
    int n = index;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    // scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w, 0.);
    // scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h, 0.);

    // scalar_t bin_size_h = roi_height / pooled_height;
    // scalar_t bin_size_w = roi_width / pooled_width;
    // original
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w + 1., 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h + 1., 0.);
    scalar_t bin_size_h = roi_height / (scalar_t)pooled_height;
    scalar_t bin_size_w = roi_width / (scalar_t)pooled_width;

    int stride_w = fmaxf(1,round(bin_size_w));
    int stride_h = fmaxf(1,round(bin_size_h));
    // const scalar_t *offset_bottom_data =
    //     bottom_data + (roi_batch_ind * channels + c) * height * width;

    // int sample_num_h = (sample_num > 0)
    //                        ? sample_num
    //                        : ceil(roi_height / pooled_height);  // e.g., = 2
    // int sample_num_w =
    //     (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    scalar_t h = (scalar_t)ph * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)pw * bin_size_w + roi_start_w;

    int hstart = fminf(floor((float)(ph) * bin_size_h + roi_start_h), height - 2);
    int wstart = fminf(floor((float)(ph) * bin_size_h + roi_start_h), height - 2);

    int img_start = roi_batch_ind * channels * height * width;

    if (h < 0 || h >= height || w < 0 || w >= width) {
        top_data[index] = 0.;
    } else {
        for(int hidx=0; hidx<=stride_h; hidx+=stride_h){
            for(int widx=0; widx<=stride_w; widx+=stride_w){
                if( ((widx+wstart)>=0) && ((widx+wstart)<width) && ((hidx+hstart)>=0) && ((hidx+hstart)<height) ){
                int cur_loc = img_start + (c * height + hstart) * width + wstart + hidx*width + widx;
                scalar_t h_ratio = 1. - (scalar_t)fabsf(h-hstart-hidx)/(scalar_t)stride_h;
                scalar_t w_ratio = 1. - (scalar_t)fabsf(w-wstart-widx)/(scalar_t)stride_w;

                top_data[index]+=bottom_data[cur_loc]*h_ratio*w_ratio;
                }
            }
        }
    }
  }
}

int ROIAlignAdaForwardLaucher(const at::Tensor features, const at::Tensor rois,
                           const float spatial_scale, const int sample_num,
                           const int channels, const int height,
                           const int width, const int num_rois,
                           const int pooled_height, const int pooled_width,
                           at::Tensor output) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "ROIAlignAdaLaucherForward", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();

        ROIAlignForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sample_num, channels, height, width, pooled_height,
                pooled_width, top_data);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}

template <typename scalar_t>
__global__ void ROIAlignAdaBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *bottom_rois,
    const scalar_t spatial_scale, const int sample_num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the aligned output
    int n = index;
    int pw = n % pooled_width;
    n /= pooled_width;
    int ph = n % pooled_height;
    n /= pooled_height;
    int c = n % channels;
    n /= channels;

    const scalar_t *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_end_w = offset_bottom_rois[3] * spatial_scale;
    scalar_t roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    scalar_t roi_width = fmaxf((scalar_t)roi_end_w - roi_start_w + 1., 0.);
    scalar_t roi_height = fmaxf((scalar_t)roi_end_h - roi_start_h + 1., 0.);

    scalar_t bin_size_h = roi_height / (scalar_t)pooled_height;
    scalar_t bin_size_w = roi_width / (scalar_t)pooled_width;

    int stride_w = fmaxf(1, round(bin_size_w));
    int stride_h = fmaxf(1, round(bin_size_h));
    // scalar_t *offset_bottom_diff =
    //     bottom_diff + (roi_batch_ind * channels + c) * height * width;
    // int offset_top = (n * channels + c) * pooled_height * pooled_width +
    //                  ph * pooled_width + pw;
    // scalar_t offset_top_diff = top_diff[offset_top];

    // int sample_num_h = (sample_num > 0)
    //                        ? sample_num
    //                        : ceil(roi_height / pooled_height);  // e.g., = 2
    // int sample_num_w =
    //     (sample_num > 0) ? sample_num : ceil(roi_width / pooled_width);

    // const scalar_t count = (scalar_t)(sample_num_h * sample_num_w);

    scalar_t h = (scalar_t)ph * bin_size_h + roi_start_h;
    scalar_t w = (scalar_t)pw * bin_size_w + roi_start_w;

    int hstart = fminf(floor((float)(ph) * bin_size_h + roi_start_h), height - 2);
    int wstart = fminf(floor((float)(pw) * bin_size_w + roi_start_w), width - 2);

    int img_start = roi_batch_ind * channels * height * width;

    if (!(h < 0 || h >= height || w < 0 || w >= width)) {
                for(int hidx=0; hidx<=stride_h; hidx+=stride_h){
                    for(int widx=0; widx<=stride_w; widx+=stride_w){
                        if( ((hstart+hidx)>=0) && ((hstart+hidx)<height) && ((wstart+widx)>=0) && ((wstart+widx)<width) ){
                        int cur_loc = img_start + (c * height + hstart) * width + wstart + hidx*width + widx;
                        scalar_t h_ratio = 1. - (scalar_t)fabsf(h-hstart-hidx)/(scalar_t)(stride_h);
                        scalar_t w_ratio = 1. - (scalar_t)fabsf(w-wstart-widx)/(scalar_t)(stride_w);

                        atomicAdd(bottom_diff + cur_loc, top_diff[index]*h_ratio*w_ratio);
                        }
                    }
                }
            }
  }
}

int ROIAlignAdaBackwardLaucher(const at::Tensor top_grad, const at::Tensor rois,
                            const float spatial_scale, const int sample_num,
                            const int channels, const int height,
                            const int width, const int num_rois,
                            const int pooled_height, const int pooled_width,
                            at::Tensor bottom_grad) {
  const int output_size = num_rois * pooled_height * pooled_width * channels;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "ROIAlignAdaLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();
        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        ROIAlignBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data, spatial_scale, sample_num,
                channels, height, width, pooled_height, pooled_width,
                bottom_diff);
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}
