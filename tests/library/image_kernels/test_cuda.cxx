/* This file is part of VIAME. See LICENSE.txt for the BSD 3-Clause license. */
#include <gtest/gtest.h>
#include <image_kernels/cuda.h>
#include <stdexcept>
#include <vector>
namespace gpu = viame::image_kernels::cuda;
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
TEST(cuda, public_api_and_ownership) {
  if (!gpu::device_count())
    GTEST_SKIP() << gpu::availability_error();
  gpu::image retained;
  std::vector<float> source(8 * 12, 7.f), host(8 * 12, -1.f);
  {
    gpu::context ctx;
    auto input = ctx.allocate(9, 8, 1, gpu::pixel_type::float32);
    ctx.upload(input, source.data(), 12 * sizeof(float));
    retained = ctx.gaussian_blur(input, 21);
    ctx.gaussian_blur(retained, 5, 0, &retained);
    EXPECT_THROW(ctx.upload(input, source.data(), 8 * sizeof(float)),
                 std::invalid_argument);
    EXPECT_THROW(ctx.allocate(0, 3, 1, gpu::pixel_type::uint8),
                 std::invalid_argument);
    EXPECT_THROW(ctx.gaussian_blur(input, 2), std::invalid_argument);
  }
  gpu::context another;
  another.download(retained, host.data(), 12 * sizeof(float));
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 12; ++x)
      if (x < 9)
        EXPECT_NEAR(host[y * 12 + x], 7.f, 2e-6);
      else
        EXPECT_EQ(host[y * 12 + x], -1.f);
}
TEST(cuda, nlm_and_image_copy) {
  if (!gpu::device_count())
    GTEST_SKIP() << gpu::availability_error();
  gpu::context ctx;
  std::vector<unsigned char> source(13 * 17 * 3, 42), host(source.size());
  auto input = ctx.allocate(17, 13, 3, gpu::pixel_type::uint8);
  ctx.upload(input, source.data());
  auto shared = input;
  ctx.denoise_non_local_means(input, 3, 7, 21, &shared);
  ctx.download(input, host.data());
  EXPECT_EQ(host, source);
}
