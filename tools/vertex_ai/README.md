# Vertex AI pipeline parameters

Prediction requests can supply `frame_rate` and `calibration_file` in the
top-level `parameters` object. For example, to process stereo videos:

```json
{
  "instances": [
    {
      "input_paths": ["gs://bucket/left.mp4", "gs://bucket/right.mp4"],
      "pipeline": "pipelines/stereo_detect_and_measure_gmm_motion.pipe"
    }
  ],
  "parameters": {
    "frame_rate": "5",
    "calibration_file": "gs://bucket/calibration_matrices.npz"
  }
}
```

`calibration_file` accepts a path inside the container or a GCS URI. GCS
calibration is downloaded before the pipeline runs. Use a calibration format
supported by the selected pipeline.

Set `VIAME_CALIBRATION_FILE` when starting either Vertex AI container to provide
a default, just as `VIAME_FRAME_RATE` provides the default frame rate. Request
parameters override these defaults. Calibration defaults to empty, which leaves
the pipeline's calibration settings unchanged; an empty request value also
disables the container default for that request.

The calibration path is passed to `measurer:calibration_file`,
`stereo_pairing:calibration_file`, and
`depth_map:computer:ocv_stereo_disparity:calibration_file`, covering the measurement,
pairing, and disparity stages used by the stereo pipelines. Per-instance
`settings` are applied last and can override individual calibration settings.
