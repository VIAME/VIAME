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

/// Lower the budget for the rest of the process, or restore it with 0.
///
/// `cv2.setNumThreads` is why this exists: the code a dataloader worker runs
/// calls it with 0 or 1 so that a library does not spawn a thread per core
/// inside each of eight workers. It can only **reduce** the budget -- the
/// pool is built once, at the size the environment asked for, and handing out
/// more jobs than it has workers is not a thing to make possible for the
/// convenience of a knob nobody turns upwards.
VIAME_IMAGE_KERNELS_EXPORT void set_kernel_thread_count ( std::size_t count );
VIAME_IMAGE_KERNELS_EXPORT void
parallel_rows ( std::size_t begin, std::size_t end, std::size_t grain,
                std::function<void ( std::size_t, std::size_t )> const &work );

} // namespace image_kernels
} // namespace viame
#endif
