// Compatibility include for the optional CUDA implementation.
#ifndef VIAME_CUDA_RESIZE_PLAN_H
#define VIAME_CUDA_RESIZE_PLAN_H
#include "letterbox_plan.h"
namespace viame { namespace image_kernels { namespace cuda { namespace detail {
using ::viame::image_kernels::detail::make_letterbox_plan;
} } } }
#endif
