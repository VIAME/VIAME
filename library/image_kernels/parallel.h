/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_KERNELS_PARALLEL_H
#define VIAME_IMAGE_KERNELS_PARALLEL_H

#include "viame_image_kernels_export.h"
#include <cstddef>
#include <functional>

namespace viame
{
namespace image_kernels
{

/// Process-wide worker budget, read once from VIAME_NUM_THREADS. Defaults to
/// min(4, hardware concurrency); 1 disables internal parallelism. Set before
/// the first kernel call. Nested work executes serially on its current worker.
VIAME_IMAGE_KERNELS_EXPORT std::size_t kernel_thread_count ();
VIAME_IMAGE_KERNELS_EXPORT void
parallel_rows ( std::size_t begin, std::size_t end, std::size_t grain,
                std::function<void ( std::size_t, std::size_t )> const &work );

} // namespace image_kernels
} // namespace viame
#endif
