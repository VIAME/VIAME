Embedding normal pipelines from C++
==================================

`viame::prepare_embedded_pipeline` converts a normal `.pipe` into an adapter
pipeline. It is implemented in `plugins/core/embedded_pipeline.cxx` on main
and `library/utilities/embedded_pipeline.cxx` on lite. Python's
`viame.open(..., embedded=True)` calls the same implementation.

The C++ API has no Python dependency. Pipelines that contain Python algorithms
still need their usual Python runtime and plugins. This API accepts `.pipe`
files; ZIP extraction and model wrapping remain responsibilities of the
higher-level Python loader.

On main, include `<viame/embedded_pipeline.h>` and link
`viame_embedded_pipeline`. Inside a VIAME CMake build:

```cmake
target_link_libraries(my_app PRIVATE viame_embedded_pipeline)
```

On lite, include `<viame/utilities/embedded_pipeline.h>`. The in-tree target
has the same name and is included in the combined `viame` library.

Prepare and build
-----------------

```cpp
#include <viame/embedded_pipeline.h>
#include <sprokit/processes/adapters/embedded_pipeline.h>

viame::embedded_pipeline_options options;
options.search_paths = { "/opt/viame/configs/pipelines" };
// Optional overrides for custom source/sink process types:
// options.inputs = std::vector<std::string>{ "input1", "input2" };
// options.outputs = std::vector<std::string>{ "detector_writer" };

auto description = viame::prepare_embedded_pipeline("detector.pipe", options);
kwiver::embedded_pipeline pipeline;
description.build(pipeline);
pipeline.start();
```

Constructing the native `embedded_pipeline` automatically loads its plugins.
The plugin manager skips registration already completed, including after an
explicit `load_all_plugins()` call. No manual initialization call is needed.

Preparation parses includes and resolves configuration and relative model
paths. It does not instantiate algorithms or load models. `build` configures
a fresh native embedded pipeline directly from memory; no generated pipeline
file is needed. `pipeline_text` is also available for inspection or saving.
Source/model assets must remain available for the pipeline's lifetime.

By default, video readers and standard output writers are replaced. Custom
process names can be specified in `options.inputs` and `options.outputs`.
An unspecified selection triggers automatic detection; an explicitly empty
selection is an error. Selected readers must be sources and selected writers
must be sinks. Other processes keep their original behavior.

Send and receive
----------------

`input_names` lists the original reader names. `input_ports` and
`output_ports` map original `process.port` names to the generated adapter
port names. Populate **every connected input port** with its native type.
For a typical single-camera detector with a downsampler:

```cpp
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/timestamp.h>

// image is an existing kwiver::vital::image_container_sptr.
auto data = kwiver::adapter::adapter_data_set::create();
data->add_value<kwiver::vital::image_container_sptr>(
    description.input_ports.at("input.image"), image);

kwiver::vital::timestamp timestamp;
timestamp.set_frame(1);
timestamp.set_time_seconds(0.0);
data->add_value(description.input_ports.at("input.timestamp"), timestamp);
data->add_value<std::string>(description.input_ports.at("input.file_name"), "frame00001");
data->add_value<double>(description.input_ports.at("input.frame_rate"), 1.0);
pipeline.send(data);

auto result = pipeline.receive();
if (!result->is_end_of_data())
{
  auto detections = result->value<kwiver::vital::detected_object_set_sptr>(
      description.output_ports.at("detector_writer.detected_object_set"));
  // Use detections here.
}

pipeline.send_end_of_input();
while (!pipeline.at_end())
{
  pipeline.receive(); // Drain remaining results and the end-of-data marker.
}
pipeline.wait();
```

For stereo or additional cameras, populate the mapped `input1.image`,
`input2.image`, etc. ports in the same data set, together with whichever
metadata ports are connected. Use base `image_container_sptr` pointers for
image datums and `double` for frame rates; the datum's static C++ type matters.

Sampling and batching are preserved, so not every input produces an output.
Output branches must have compatible rates because one output adapter
synchronizes their ports. Adapter queues are bounded: interleave sending and
receiving, and drain outputs when finishing. Native `empty()` can be used
before `receive()` when polling is appropriate.

Lite uses `viame::embedded_pipeline`, `viame::adapter::adapter_data_set` and
`viame::image_container_sptr`/`viame::timestamp` in place of the corresponding
KWIVER namespaces. Its native adapter header is
`<viame/pipeline_framework/adapters/embedded_pipeline.h>`.

Python binding
--------------

The binding lives beside the C++ implementation in
`plugins/core/embedded_pipeline_python.cxx` on main. The direct entry point is:

```python
from viame.core.embedded_pipeline import prepare_embedded_pipeline

description = prepare_embedded_pipeline(
    "detector.pipe", search_paths=["/opt/viame/configs/pipelines"])
# description.build(native_embedded_pipeline)
```

On lite it is `viame.utilities.embedded_pipeline`. Most Python callers should
continue using `viame.open(..., embedded=True)`, which adds image conversion,
metadata defaults, receive timeouts, and cleanup around this shared C++ API.
