# Plugin Creation

This document corresponds to the [Plugin Creation](https://github.com/VIAME/VIAME/tree/main/examples/plugin_creation) example in a VIAME desktop installation.

## External Plugin Creation

This directory contains the source files needed to make a loadable algorithm plugin implementation external to VIAME, which links against an installation, or in the case of python generates a loadable script. This is for cases where we might want to just make a plugin against pre-compiled binaries, instead of building all of VIAME itself.

The procedure is slightly different depending on whether you are developing an external C++ or Python module. C++ modules require linking your module against VIAME, the output of which is plugin DLL which can be used directly in VIAME pipelines. Python processes can be made without compilation, and placed in your PYTHONPATH for use by the plugin system.

## Internal Plugin Creation

A plugin can also be added inside the VIAME source tree and built with the rest of VIAME. Templates for a new detector are provided in `[viame-source]/plugins/templates`, with one folder each for C++, Python and Matlab.

### Starting from the C++ template

A detector is a class that implements the `kwiver::vital::algo::image_object_detector` interface.

1. Copy the contents of `plugins/templates/cxx` to a new folder under `plugins`.
2. Rename `template_detector.cxx` and `template_detector.h` after the new detector. `CMakeLists.txt` and `register_algorithms.cxx` keep their names.
3. Replace the placeholders in every file:

| Placeholder | Replace with |
|----|----|
| `@template@` | Name of the detector |
| `@template_lib@` | Name of the plugin library that will hold the detector. It can match the detector name. |
| `@template_dir@` | Name of the new source folder. For `plugins/fin_fish_detector` this is `fin_fish_detector`. |

The placeholders also appear in capital letters, where the replacement should be capitalized too.

### Filling in the detector

The template has four methods to complete. `get_configuration` and `set_configuration` declare and read the detector's settings, `check_configuration` reports anything that would stop it running, and `detect` does the work.

Most of the work in `detect` is converting the input image to the form the detector needs, and converting its results back. Many detectors take an OpenCV image, which is obtained from the input as follows:

```cpp
// image_data is the kwiver::vital::image_container_sptr passed to detect()
cv::Mat cv_image = kwiver::arrows::ocv::image_container::vital_to_ocv(
  image_data->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
```

Detectors usually return boxes, each with one or more class labels and scores. These are returned as a detected object set:

```cpp
auto detected_set = std::make_shared< kwiver::vital::detected_object_set >();

for( const auto& result : results )
{
  // Coordinates are given in the order left, top, right, bottom
  kwiver::vital::bounding_box_d bbox(
    result.left, result.top, result.right, result.bottom );

  // Holds every class label for this box with its score
  auto type = std::make_shared< kwiver::vital::detected_object_type >();

  for( const auto& label : result.labels )
  {
    type->set_score( label.name, label.score );
  }

  detected_set->add(
    std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, type ) );
}

return detected_set;
```

Here `results` stands for whatever the detector produced. Once built, the detector is selected in a pipeline file by the name given to `@template@`.

## DIVE Documentation

- [DIVE pipeline import and export](https://viame.github.io/VIAME/sections/dive/Pipeline-Import-Export.html) covers loading a custom pipeline into the interface, which is how a new plugin is run from DIVE

## Code and Build Flags

Flags to enable when building VIAME from source for this example:

- `VIAME_ENABLE_PYTHON`

Source code:

- examples/plugin_creation/cxx/example_detector.cxx
- examples/plugin_creation/cxx/example_detector.h
- examples/plugin_creation/cxx/register_algorithms.cxx
- examples/plugin_creation/python/example_filter.py
- examples/plugin_creation/python/example_filter_process.py
- plugins/templates/cxx/template_detector.cxx
- plugins/templates/cxx/template_detector.h
- plugins/templates/python/template_detector.py
