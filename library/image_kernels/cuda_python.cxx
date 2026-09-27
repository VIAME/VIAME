/* This file is part of VIAME. See LICENSE.txt for the BSD 3-Clause license. */
#include "cuda.h"
#include <climits>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
namespace py = pybind11;
namespace gpu = viame::image_kernels::cuda;
namespace {
py::tuple shape(gpu::image const &image) {
  if (image.channels() == 1)
    return py::make_tuple(image.height(), image.width());
  return py::make_tuple(image.height(), image.width(), image.channels());
}
py::dtype dtype(gpu::image const &image) {
  return image.type() == gpu::pixel_type::uint8 ? py::dtype::of<unsigned char>()
                                                : py::dtype::of<float>();
}
} // namespace
PYBIND11_MODULE(_cuda, m) {
  m.doc() = "Optional CUDA image kernels; calls complete before returning.";
  m.def("device_count", &gpu::device_count);
  m.def("availability_error", &gpu::availability_error);
  py::class_<gpu::image>(m, "Image")
      .def_property_readonly("shape", &shape)
      .def_property_readonly("dtype", &dtype)
      .def_property_readonly("device", &gpu::image::device);
  py::class_<gpu::context>(m, "Context")
      .def(py::init<int>(), py::arg("device") = 0,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "upload",
          [](gpu::context &ctx, py::array const &source,
             gpu::image const *output) {
            if (source.ndim() != 2 && source.ndim() != 3)
              throw py::value_error("CUDA upload expects HxW or HxWxC");
            for (int axis = 0; axis < source.ndim(); ++axis)
              if (source.shape(axis) <= 0 || source.shape(axis) > INT_MAX / 4)
                throw py::value_error("invalid CUDA image dimensions");
            gpu::pixel_type type;
            if (source.dtype().is(py::dtype::of<unsigned char>()))
              type = gpu::pixel_type::uint8;
            else if (source.dtype().is(py::dtype::of<float>()))
              type = gpu::pixel_type::float32;
            else
              throw py::type_error(
                  "CUDA upload supports uint8 and float32 only");
            auto packed = py::array::ensure(source, py::array::c_style);
            if (!packed)
              throw py::value_error("cannot make a contiguous image");
            int w = static_cast<int>(source.shape(1)),
                h = static_cast<int>(source.shape(0));
            int c = source.ndim() == 3 ? static_cast<int>(source.shape(2)) : 1;
            if (output && (output->width() != w || output->height() != h ||
                           output->channels() != c || output->type() != type))
              throw py::value_error(
                  "CUDA upload output must match input shape and dtype");
            auto pointer = packed.data();
            py::gil_scoped_release release;
            auto result = output ? *output : ctx.allocate(w, h, c, type);
            ctx.upload(result, pointer);
            return result;
          },
          py::arg("image"), py::arg("out") = nullptr)
      .def(
          "download",
          [](gpu::context &ctx, gpu::image const &image) {
            std::vector<py::ssize_t> dimensions{image.height(), image.width()};
            if (image.channels() != 1)
              dimensions.push_back(image.channels());
            auto result = py::array(dtype(image), dimensions);
            auto pointer = result.mutable_data();
            {
              py::gil_scoped_release release;
              ctx.download(image, pointer);
            }
            return result;
          },
          py::arg("image"))
      .def("gaussian_blur", &gpu::context::gaussian_blur, py::arg("image"),
           py::arg("size"), py::arg("sigma") = 0., py::arg("out") = nullptr,
           py::call_guard<py::gil_scoped_release>())
      .def("denoise_non_local_means", &gpu::context::denoise_non_local_means,
           py::arg("image"), py::arg("strength"), py::arg("patch") = 7,
           py::arg("window") = 21, py::arg("out") = nullptr,
           py::call_guard<py::gil_scoped_release>());
}
