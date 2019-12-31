#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "roi_align_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)


    __global__ void ROIAlignForward(const int nthreads, const float* bottom_data, const float spatial_scale, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ph, pw) is an element in the aligned output
            int n = index;
            int pw = n % aligned_width;
            n /= aligned_width;
            int ph = n % aligned_height;
            n /= aligned_height;
            int c = n % channels;
            n /= channels;

            bottom_rois += n * 5;
            float roi_batch_ind = bottom_rois[0];
            float roi_start_w = bottom_rois[1] * spatial_scale;
            float roi_start_h = bottom_rois[2] * spatial_scale;
            float roi_end_w = bottom_rois[3] * spatial_scale;
            float roi_end_h = bottom_rois[4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);

            /* original code */

            float bin_size_h = roi_height / (aligned_height - 1.);
            float bin_size_w = roi_width / (aligned_width - 1.);

            float h = (float)(ph) * bin_size_h + roi_start_h;
            float w = (float)(pw) * bin_size_w + roi_start_w;


            /*======================from ilchae========================*/
            /*
            float bin_size_h = roi_height / (aligned_height + 1.);
            float bin_size_w = roi_width / (aligned_width + 1.);

            float h = (float)(ph+1) * bin_size_h + roi_start_h;
            float w = (float)(pw+1) * bin_size_w + roi_start_w;
            */
            ////////////////////////////////////////////////////////////

            int hstart = fminf(floor(h), height - 2);
            int wstart = fminf(floor(w), width - 2);


            int img_start = roi_batch_ind * channels * height * width;

            // bilinear interpolation
            if (h < 0 || h >= height || w < 0 || w >= width) {
                top_data[index] = 0.;
            } else {
                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);
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


    int ROIAlignForwardLaucher(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;


        ROIAlignForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, bottom_data, spatial_scale, height, width, channels, aligned_height, aligned_width, bottom_rois, top_data);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }


    __global__ void ROIAlignBackward(const int nthreads, const float* top_diff, const float spatial_scale, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, float* bottom_diff, const float* bottom_rois) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

            // (n, c, ph, pw) is an element in the aligned output
            int n = index;
            int pw = n % aligned_width;
            n /= aligned_width;
            int ph = n % aligned_height;
            n /= aligned_height;
            int c = n % channels;
            n /= channels;

            bottom_rois += n * 5;
            float roi_batch_ind = bottom_rois[0];
            float roi_start_w = bottom_rois[1] * spatial_scale;
            float roi_start_h = bottom_rois[2] * spatial_scale;
            float roi_end_w = bottom_rois[3] * spatial_scale;
            float roi_end_h = bottom_rois[4] * spatial_scale;
            /* int roi_start_w = round(bottom_rois[1] * spatial_scale); */
            /* int roi_start_h = round(bottom_rois[2] * spatial_scale); */
            /* int roi_end_w = round(bottom_rois[3] * spatial_scale); */
            /* int roi_end_h = round(bottom_rois[4] * spatial_scale); */

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
            /* ============ original code =========== */

            float bin_size_h = roi_height / (aligned_height - 1.);
            float bin_size_w = roi_width / (aligned_width - 1.);

            float h = (float)(ph) * bin_size_h + roi_start_h;
            float w = (float)(pw) * bin_size_w + roi_start_w;


            /*======================from ilchae========================*/
            /*
            float bin_size_h = roi_height / (aligned_height + 1.);
            float bin_size_w = roi_width / (aligned_width + 1.);

            float h = (float)(ph+1) * bin_size_h + roi_start_h;
            float w = (float)(pw+1) * bin_size_w + roi_start_w;
            */
            ////////////////////////////////////////////////////////////


            int hstart = fminf(floor(h), height - 2);
            int wstart = fminf(floor(w), width - 2);

            int img_start = roi_batch_ind * channels * height * width;

            // bilinear interpolation
            if (!(h < 0 || h >= height || w < 0 || w >= width)) {
                float h_ratio = h - (float)(hstart);
                float w_ratio = w - (float)(wstart);
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

    int ROIAlignBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* bottom_diff, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;

        ROIAlignBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, top_diff, spatial_scale, height, width, channels, aligned_height, aligned_width, bottom_diff, bottom_rois);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }

    __global__ void ROIAlignAdaForward(const int nthreads, const float* bottom_data, const float spatial_scale, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ph, pw) is an element in the aligned output
            int n = index;
            int pw = n % aligned_width;
            n /= aligned_width;
            int ph = n % aligned_height;
            n /= aligned_height;
            int c = n % channels;
            n /= channels;

            bottom_rois += n * 5;
            float roi_batch_ind = bottom_rois[0];
            float roi_start_w = bottom_rois[1] * spatial_scale;
            float roi_start_h = bottom_rois[2] * spatial_scale;
            float roi_end_w = bottom_rois[3] * spatial_scale;
            float roi_end_h = bottom_rois[4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);

            /* original code */

            float bin_size_h = roi_height / (float)(aligned_height);
            float bin_size_w = roi_width / (float)(aligned_width);

            int stride_w = fmaxf(1,round(bin_size_w));
            int stride_h = fmaxf(1,round(bin_size_h));


            float h = (float)(ph) * bin_size_h + roi_start_h; // this is right in geometically
            float w = (float)(pw) * bin_size_w + roi_start_w; // this is right in geometically




            int hstart = fminf(floor((float)(ph) * bin_size_h + roi_start_h), height - 2);
            int wstart = fminf(floor((float)(pw) * bin_size_w + roi_start_w), width - 2);

            int img_start = roi_batch_ind * channels * height * width;

            // bilinear interpolation
            if (h < 0 || h >= height || w < 0 || w >= width) {
                top_data[index] = 0.;
            } else {
                for(int hidx=0; hidx<=stride_h; hidx+=stride_h){
                    for(int widx=0; widx<=stride_w; widx+=stride_w){
                        if( ((widx+wstart)>=0) && ((widx+wstart)<width) && ((hidx+hstart)>=0) && ((hidx+hstart)<height) ){
                        int cur_loc = img_start + (c * height + hstart) * width + wstart + hidx*width + widx;
                        float h_ratio = 1. - (float)fabsf(h-hstart-hidx)/(float)stride_h;
                        float w_ratio = 1. - (float)fabsf(w-wstart-widx)/(float)stride_w;

                        top_data[index]+=bottom_data[cur_loc]*h_ratio*w_ratio;
                        }
                    }
                }
            }
        }
    }


    int ROIAlignAdaForwardLaucher(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;


        ROIAlignAdaForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, bottom_data, spatial_scale, height, width, channels, aligned_height, aligned_width, bottom_rois, top_data);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }


    __global__ void ROIAlignAdaBackward(const int nthreads, const float* top_diff, const float spatial_scale, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, float* bottom_diff, const float* bottom_rois) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

            // (n, c, ph, pw) is an element in the aligned output
            int n = index;
            int pw = n % aligned_width;
            n /= aligned_width;
            int ph = n % aligned_height;
            n /= aligned_height;
            int c = n % channels;
            n /= channels;

            bottom_rois += n * 5;
            float roi_batch_ind = bottom_rois[0];
            float roi_start_w = bottom_rois[1] * spatial_scale;
            float roi_start_h = bottom_rois[2] * spatial_scale;
            float roi_end_w = bottom_rois[3] * spatial_scale;
            float roi_end_h = bottom_rois[4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
            /* ============ original code =========== */

            float bin_size_h = roi_height / (float)(aligned_height);
            float bin_size_w = roi_width / (float)(aligned_width);

            int stride_w = fmaxf(1,round(bin_size_w));
            int stride_h = fmaxf(1,round(bin_size_h));

            float h = (float)(ph) * bin_size_h + roi_start_h;
            float w = (float)(pw) * bin_size_w + roi_start_w;

            int hstart = fminf(floor((float)(ph) * bin_size_h + roi_start_h), height - 2);
            int wstart = fminf(floor((float)(pw) * bin_size_w + roi_start_w), width - 2);

            int img_start = roi_batch_ind * channels * height * width;

            // bilinear interpolation
            if (!(h < 0 || h >= height || w < 0 || w >= width)) {
                for(int hidx=0; hidx<=stride_h; hidx+=stride_h){
                    for(int widx=0; widx<=stride_w; widx+=stride_w){
                        if( ((hstart+hidx)>=0) && ((hstart+hidx)<height) && ((wstart+widx)>=0) && ((wstart+widx)<width) ){
                        int cur_loc = img_start + (c * height + hstart) * width + wstart + hidx*width + widx;
                        float h_ratio = 1. - (float)fabsf(h-hstart-hidx)/(float)(stride_h);
                        float w_ratio = 1. - (float)fabsf(w-wstart-widx)/(float)(stride_w);

                        atomicAdd(bottom_diff + cur_loc, top_diff[index]*h_ratio*w_ratio);
                        }
                    }
                }
            }
        }
    }

    int ROIAlignAdaBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* bottom_diff, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;

        ROIAlignAdaBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, top_diff, spatial_scale, height, width, channels, aligned_height, aligned_width, bottom_diff, bottom_rois);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }

















    __global__ void ROIAlignDenseAdaForward(const int nthreads, const float* bottom_data, const float spatial_scale, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ph, pw) is an element in the aligned output
            int n = index;
            int pw = n % aligned_width;
            n /= aligned_width;
            int ph = n % aligned_height;
            n /= aligned_height;
            int c = n % channels;
            n /= channels;

            bottom_rois += n * 5;
            float roi_batch_ind = bottom_rois[0];
            float roi_start_w = bottom_rois[1] * spatial_scale;
            float roi_start_h = bottom_rois[2] * spatial_scale;
            float roi_end_w = bottom_rois[3] * spatial_scale;
            float roi_end_h = bottom_rois[4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);

            /* original code */

            float bin_size_h = roi_height / (float)(aligned_height);
            float bin_size_w = roi_width / (float)(aligned_width);

            int stride_w = fmaxf(1,round(bin_size_w));
            int stride_h = fmaxf(1,round(bin_size_h));


            float h = (float)(ph) * bin_size_h + roi_start_h; // this is right in geometically
            float w = (float)(pw) * bin_size_w + roi_start_w; // this is right in geometically




            int hstart = fminf(floor((float)(ph) * bin_size_h + roi_start_h), height - 2);
            int wstart = fminf(floor((float)(pw) * bin_size_w + roi_start_w), width - 2);

            int img_start = roi_batch_ind * channels * height * width;

            // bilinear interpolation
            if (h < 0 || h >= height || w < 0 || w >= width) {
                top_data[index] = 0.;
            } else {

                float ratio_sum = 0. ;
                for(int hidx=0; hidx<=stride_h; hidx++){
                    for(int widx=0; widx<=stride_w; widx++){
                        int cur_loc = img_start + (c * height + hstart) * width + wstart + hidx*width + widx;
                        float h_ratio = 1. - (float)fabsf(h-hstart-hidx)/(float)stride_h;
                        float w_ratio = 1. - (float)fabsf(w-wstart-widx)/(float)stride_w;

                        float ratio = h_ratio * w_ratio;
                        ratio_sum += ratio;
                        top_data[index]+=bottom_data[cur_loc]*ratio;
                    }
                }
                top_data[index]/=ratio_sum;
            }
        }
    }


    int ROIAlignDenseAdaForwardLaucher(const float* bottom_data, const float spatial_scale, const int num_rois, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* top_data, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;


        ROIAlignDenseAdaForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, bottom_data, spatial_scale, height, width, channels, aligned_height, aligned_width, bottom_rois, top_data);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }


    __global__ void ROIAlignDenseAdaBackward(const int nthreads, const float* top_diff, const float spatial_scale, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, float* bottom_diff, const float* bottom_rois) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {

            // (n, c, ph, pw) is an element in the aligned output
            int n = index;
            int pw = n % aligned_width;
            n /= aligned_width;
            int ph = n % aligned_height;
            n /= aligned_height;
            int c = n % channels;
            n /= channels;

            bottom_rois += n * 5;
            float roi_batch_ind = bottom_rois[0];
            float roi_start_w = bottom_rois[1] * spatial_scale;
            float roi_start_h = bottom_rois[2] * spatial_scale;
            float roi_end_w = bottom_rois[3] * spatial_scale;
            float roi_end_h = bottom_rois[4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w + 1., 0.);
            float roi_height = fmaxf(roi_end_h - roi_start_h + 1., 0.);
            /* ============ original code =========== */

            float bin_size_h = roi_height / (float)(aligned_height);
            float bin_size_w = roi_width / (float)(aligned_width);

            int stride_w = fmaxf(1,round(bin_size_w));
            int stride_h = fmaxf(1,round(bin_size_h));

            float h = (float)(ph) * bin_size_h + roi_start_h;
            float w = (float)(pw) * bin_size_w + roi_start_w;

            int hstart = fminf(floor((float)(ph) * bin_size_h + roi_start_h), height - 2);
            int wstart = fminf(floor((float)(pw) * bin_size_w + roi_start_w), width - 2);

            int img_start = roi_batch_ind * channels * height * width;

            // bilinear interpolation
            if (!(h < 0 || h >= height || w < 0 || w >= width)) {
                for(int hidx=0; hidx<=stride_h; hidx+=stride_h){
                    for(int widx=0; widx<=stride_w; widx+=stride_w){
                        int cur_loc = img_start + (c * height + hstart) * width + wstart + hidx*width + widx;

                        //float h_ratio = 1. - (float)fabsf(h-hstart-hidx)/(float)(stride_h);
                        //float w_ratio = 1. - (float)fabsf(w-wstart-widx)/(float)(stride_w);

                        float ratio = 1. / (2.505*(stride_h+1.)*(stride_w+1.)) * expf( -0.5*(powf((h-hstart-hidx)/(float)stride_h,2.) + powf( (w-wstart-widx)/(float)stride_w, 2.)) ) ;

                        atomicAdd(bottom_diff + cur_loc, top_diff[index]*ratio);
                    }
                }
            }
        }
    }

    int ROIAlignDenseAdaBackwardLaucher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width, const int channels, const int aligned_height, const int aligned_width, const float* bottom_rois, float* bottom_diff, cudaStream_t stream) {
        const int kThreadsPerBlock = 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;

        ROIAlignDenseAdaBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(output_size, top_diff, spatial_scale, height, width, channels, aligned_height, aligned_width, bottom_diff, bottom_rois);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }



#ifdef __cplusplus
}
#endif


