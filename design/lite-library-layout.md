# VIAME lite: library layout and file mapping

## 1. Target tree

```
viame/
  CMakeLists.txt  CMakePresets.json  cmake/
  third_party/            vendored sources only (lite-dependencies.md §4)
  library/
    core_types/           image, image_container, bounding_box, detected_object(+set,+type),
                          track(+set), object_track_set, timestamp, descriptor(+set), point,
                          polygon, homography, transform_2d, camera(+intrinsics,+perspective,
                          +rig), landmark, feature(+set,+track_set), match_set, uid,
                          category_hierarchy, database_query / query_result / iqr_feedback,
                          math/ (vector, matrix, rotation, covariance, decomp, similarity)
    algorithm_framework/  config_block (+parser, +io), logger, exceptions, registry (static
                          registration, aliases, lazy python factories, optional external-plugin
                          hook), algorithm base + macros, algorithm interfaces (algo/), applet
                          base, python bindings for this + core_types
    pipeline_framework/   process, port, edge, datum, stamp, pipeline, schedulers, .pipe parser
                          and bakery, cluster, embedded_pipeline, input/output adapters, runner
                          applet, generic processes (downsample, frame filters), python process
                          API + pythread scheduler
    utilities/            file/path/glob/temp-dir helpers, string utils, manipulate_pipelines,
                          python_script_applet, compat.py
    image_ops/            pixel-type-generic kernels on core_types::image: convert, channel,
                          resample, warp, color (+demosaic), filter, binary/morphology, contours,
                          hist (+CLAHE), match (template NCC), draw (+bitmap font)
    video_io/             codecs/ (stb, tiff), image_io impls, image_list video_input, python
                          PyAV video_input/output, image_writer/video_input/video_output/
                          image_file_reader processes, filename->timestamp, shot-break detection
    file_io/              viame_csv, kw18, coco, dive, cvat, habcam, oceaneyes, fishnet, yolo,
                          transform_2d io (homography json), camera io (krtd, opencv yaml, rig),
                          descriptor csv store, database/ (cppdb, optional), json helper,
                          homography writer, detected_object_input/output, read/write_object_track,
                          write_track_descriptor processes
    image_processing/     image_filter impls (convert_image, average_frames, morphology, threshold,
                          color_commonality, white_balance, enhancer, percentile eq, debayer, color
                          correction, hue shift, convert_color), split/merge images, warp_image,
                          stabilisation, registration/alignment, optical flow, mosaics, blackout,
                          image_filter/split_image/merge_images/stabilize_image processes
    object_detectors/     image_object_detector impls (C++ and python), windowed chipping,
                          detector processes (image_object_detector, detect_motion,
                          detect_in_subregions), vertex-ai detector
    object_trackers/      track_objects impls, initialise/associate impls, kalman common code,
                          track processes (track_objects, initialize_object_tracks,
                          associate_detections_to_tracks, compute_association_matrix,
                          merge_track_sets, convert_tracks_to_detections, unwrap_detections,
                          track_conductor, accumulate/filter/resample tracks)
    classifiers/          refine_detections impls, refine_tracks impls, merge_detections impls,
                          detected_object_filter impls, full-frame classifiers, their processes
    segmentation/         sam2/sam3 segmenters and refiners, watershed/grabcut, keypoints from
                          mask, interactive segmentation service host, polygon/mask utilities
    descriptors/          compute_track_descriptors impls, IQR sessions (base, svm, adaboost),
                          query processes, descriptor ingest/fetch, augmentation, srm formatting
    measurement/          stereo calibration/rectification/disparity (python), pairing of stereo
                          detections/tracks, length measurement, triangulation, camera rigs,
                          interactive stereo, epipolar matchers, foundation stereo, colmap/SfM,
                          seagis, compute_stereo_depth_map process
    training/             train_detector/train_tracker impls, adaptive trainers, kwcoco base,
                          per-backend trainers and launchers, netharn, training data prep,
                          onnx export, train_supervisor, train_detector process
    evaluation/           evaluate_models, scoring, metric plots (python)
    examples/             hello_world detector/filter C++ and python, plugin templates
  python/                 `viame` package root, `kwiver` shim (P11), requirements/
  tools/                  viame.cxx, applets (csv, json, score, train, get_configs,
                          resample_tracks, registry-dump, pipe-check), python scripts
  configs/  examples/  tests/  docs/  design/  docker/
```

Each `library/<dir>/`:

```
CMakeLists.txt   viame_add_library(<dir> ...), per-file gating via viame_add_sources
register.cxx     void register_<dir>(viame::registry&): algorithms, processes, applets, aliases;
                 called from the generated register_builtins() (a dlopen module only in P2-P7)
*.cxx *.h        C++ (processes keep the _process suffix)
*.py             alongside the C++, installed as viame.<dir>; __init__.py declares
                 implementations lazily as (interface, name, "viame.<dir>.module:Class")
                 without importing them
tests/           unit + golden tests for this library
```

Dependency DAG (edges point at what may be linked):

```
core_types <- algorithm_framework <- pipeline_framework <- utilities <- image_ops
  <- {video_io, file_io} <- image_processing
  <- {object_detectors, object_trackers, classifiers, segmentation, descriptors, measurement}
  <- training <- evaluation
```

A library needing a sibling's helper moves the helper down, never links
sideways.

## 2. Mapping: kwiver code -> library (P5)

| Destination | Source |
|---|---|
| `core_types/` | `vital/types/*` pruned (list in `tasks/phase-05-import-kwiver.md`), `vital/vital_types.h` |
| `algorithm_framework/config`, `logger`, `exceptions`, `util`, `range`, `io`, `plugin`, `algo`, `applets` | `vital/config`, `vital/logger`+`logger_plugins`, `vital/exceptions`, `vital/util`, `vital/range`, `vital/io`, `vital/plugin_management`, `vital/algo`, `vital/applets`+`applets_plugins` |
| `algorithm_framework/`, `core_types/` (bindings alongside the C++) | `python/kwiver/vital/{types,algo(generated, replaced in P8),config,util,modules,plugin_management}` |
| `pipeline_framework/` | `sprokit/src/sprokit/pipeline`, `pipeline_util`, `sprokit/src/schedulers`, `sprokit/src/applets/pipeline_runner`, `sprokit/processes/kwiver_type_traits.h` (-> `type_traits.h`), `trait_utils.h`, `sprokit/processes/adapters/*`, `sprokit/processes/core/downsample_process` |
| `pipeline_framework/` (bindings alongside the C++) | `python/kwiver/sprokit/{pipeline,pipeline_util,processes/kwiver_process,schedulers/pythread_per_process,adapters,util}` |
| `video_io/` | `arrows/core/video_input_image_list`, `arrows/ocv/image_io` (until P7), `sprokit/processes/core/{video_input,video_output,image_writer,image_file_reader,frame_list}_process`, `sprokit/processes/ocv/image_viewer_process` (until P7) |
| `file_io/` | `arrows/core/{detected_object_set_input_kw18,detected_object_set_output_kw18,read_object_track_set_kw18,write_object_track_set_kw18,write_track_descriptor_set_csv,feature_descriptor_io}`, `sprokit/processes/core/{detected_object_input,detected_object_output,read_object_track,write_object_track,write_track_descriptor}_process`, `python/kwiver/.../homography_writer.py` |
| `image_processing/` | `arrows/ocv/{merge_images,split_image,draw_detected_object_set,refine_detections_write_to_disk,estimate_homography,estimate_fundamental_matrix,detect_features_SIFT,extract_descriptors_SIFT,detect_features_SURF,extract_descriptors_SURF,match_features_*}` and `arrows/core/{compute_ref_homography_core,track_features_core,match_features_homography,close_loops_bad_frames_only,close_loops_appearance_indexed,detect_features_filtered,filter_features_nonmax,filter_tracks,track_set_impl,match_matrix}` (all until P7), `sprokit/processes/core/{image_filter,split_image,merge_images,stabilize_image,draw_detected_object_set}_process` |
| `object_detectors/` | `arrows/ocv/{hough_circle_detector,detect_heat_map,detect_motion_3frame_differencing}` (until P7), `arrows/core/example_detector`, `sprokit/processes/core/{image_object_detector,detect_motion}_process` |
| `object_trackers/` | `arrows/core/{initialize_object_tracks_threshold,associate_detections_to_tracks_threshold}`, `sprokit/processes/core/{track_objects,initialize_object_tracks,merge_track_sets,convert_tracks_to_detections,unwrap_detections}_process` |
| `classifiers/` | `arrows/core/class_probability_filter`, `sprokit/processes/core/{refine_detections,refine_tracks,merge_detection_sets,detected_object_filter}_process` |
| `descriptors/` | `arrows/core/handle_descriptor_request_core`, `sprokit/processes/core/{compute_track_descriptors,perform_query,handle_descriptor_request}_process` |
| `measurement/` | `arrows/mvg/{triangulate,epipolar_geometry}`, `arrows/ocv/camera_intrinsics` (until P7), `sprokit/processes/core/compute_stereo_depth_map_process` |
| `opencv_bridge/` (transitional, deleted in P7) | `arrows/ocv/{image_container,mat_image_memory,descriptor_set,feature_set,match_set,bounding_box}` |
| `training/` | `sprokit/processes/core/train_detector_process` |
| `third_party/` | `vital/kwiversys`, `vital/internal/cereal` (+ rapidjson), `vital/applets/cxxopts.hpp` (kwiversys and cereal deleted in P8) |

## 3. Mapping: `plugins/` -> library (P2)

### `plugins/core`

| Destination | Files |
|---|---|
| `utilities/` | `utilities_file`, `utilities_image`, `utilities_tracks`, `manipulate_pipelines`, `applet_attributes.h`, `python_script_applet`, `utils.py`, `vital_registration.py`; `plugins/compat.py`, `plugins/types.py` |
| `video_io/` | `add_timestamp_from_filename`, `filename_to_timestamp`, `write_disparity_maps`, `read_habcam_metadata_process`, `detect_shot_breaks` (+ process) |
| `file_io/` | `read_detected_object_set_{auto,cvat,dive,fishnet,habcam,oceaneyes,viame_csv,yolo}`, `read_object_track_set_{auto,dive,viame_csv}`, `write_detected_object_set_viame_csv`, `write_object_track_set_viame_csv`, `read_transform_homography_json`, `auto_detect_transform`, `convert_notes_to_attributes`, `camera_io`, `camera_rig_io`, `store_descriptors_csv`, `write_homography_list_process`; python `read/write_*_coco.py`, `utilities_coco.py` |
| `image_processing/` | `equalize_via_percentiles` (cxx + py), `convert_polygons_to_mask`, `optical_flow.py`, `stabilize_many_images.py`, `multicam_homog_mosaic.py`, `multicam_homog_blackout.py`, `align_multimodal_imagery_process`, `warp_image_process`, `warp_detections_process`, `alignment_core.py`, `align_cameras_process.py`, `accumulate_image_statistics_process`, `stack_frames_process`, `utility_processes.py` |
| `object_detectors/` | `empty_detector`, `full_frame_detector`, `windowed_detector`, `windowed_utils` |
| `object_trackers/` | `bytetrack_tracker.py`, `ocsort_tracker.py`, `simple_homog_tracker.py`, `multicam_homog_tracker.py`, `track_conductor_process`, `accumulate_object_tracks_process`, `filter_object_tracks_process`, `resample_object_tracks_process`, `split_tracks_to_feature_landmarks_process`, `merge_tracks_tube_iou.py` |
| `classifiers/` | `convert_head_tail_points`, `refine_detections_add_fixed`, `refine_detections_nms`, `refine_tracks_average_tot`, `windowed_refiner`, `merge_detections_suppress_in_regions`, `utilities_target_clfr`, `merge_detections_{nms_fusion,coverage_reinforce,simple}.py`, `detection_fusion_core.py`, `multicam_homog_det_suppressor.py` |
| `segmentation/` | `interactive_segmentation.py`, `segmentation_utils.py`, `utilities_segmentation` (RDP), `interactive_service.py` |
| `descriptors/` | `iqr_session.h`, `utilities_iqr.h`, `average_track_descriptors`, `ingest_descriptors_process`, `fetch_descriptors_process`, `object_track_descriptors_process`, `extract_desc_ids_for_training_process`, `create_database_query_process`, `select_database_query_process`, `write_query_results_as_tracks_process` |
| `measurement/` | `measurement_utilities` (+ pybind module `_measurement`), `pair_stereo_detections` (+ process), `pair_stereo_tracks`, `measure_objects_process`, `refine_measurements_process`, `calibrate_cameras_from_tracks_process`, `interactive_stereo.py`, `survey_metadata.py` |
| `training/` | `adaptive_detector_trainer`, `adaptive_tracker_trainer`, `windowed_trainer`, `utilities_training`, `training_data.py`, `tracker_param_search.py`, `bytetrack_trainer.py`, `ocsort_trainer.py`, `frame_diff_trainer.py` |
| `evaluation/` | `evaluate_models` |
| `pipeline_framework/processes` | `image_to_image_set_process`, `filter_frame_process`, `filter_frame_index_process` |

### `plugins/opencv`

| Destination | Files |
|---|---|
| `image_processing/` | `apply_color_correction`, `convert_color_space`, `debayer_filter`, `enhance_images`, `random_hue_shift`, `split_image_habcam`, `split_image_horizontally`, `warp_image_ocv`, `fft_filter_based_on_ref.py`, `multimodal_registration.py`, `registration_utils.py` |
| `object_detectors/` | `detect_calibration_targets`, `windowed_detector` + `windowed_utils` (merged with core's in P2), `diff_of_gauss_detector`, `canny_edge_detector`, `ellipse_proposal.h`, `detect_in_subregions_process`, `stereo_processes.py::GMMDetectFishProcess`, `stereo_algos.py::GMMForegroundObjectDetector` |
| `classifiers/` | `classify_fish_hierarchical_svm`, `windowed_refiner` (merged), `refine_detections_util` |
| `segmentation/` | `add_keypoints_from_mask`, `refine_detections_grabcut`, `refine_detections_watershed`, `watershed_segmenter.py` |
| `descriptors/` | `adaboost_classifier`, `iqr_session_adaboost.h`, `gabor_features`, `hog_features`, `kmedians`, `process_query_process_adaboost` |
| `measurement/` | `calibrate_single_camera` (+ process), `calibrate_stereo_cameras`, `optimize_stereo_cameras`, `compute_stereo_disparity`, `filter_stereo_feature_tracks`, `pair_stereo_detections` (+ process, merged), `pair_stereo_tracks` (+ process, merged), `measure_objects_process` (merged), `stereo_algos.py`, `stereo_utils.py`, `stereo_pipeline.py`, `prior_coverage_opencv.py` |
| `training/` | `windowed_trainer` (merged) |
| `evaluation/` | `plot_metrics` |
| deleted | `stereo_comp.py`, `stereo_demo.py`, `stereo_expt.py`, `register_algorithms.h`, `MeasureProcess` in `stereo_processes.py` |

### `plugins/pytorch`

| Destination | Files |
|---|---|
| `object_detectors/python` | `mmdet_detector`, `netharn_detector`, `mit_yolo_detector`, `ultralytics_detector`, `rf_detr_detector`, `litdet_detector`, `detectron2_detector`, `huggingface_zeroshot_detector`, `maskcut_detector`, `remax_convnext_detector`, `remax_dino_detector`, `utilities.py` -> `base.py`, `mmdet_compatibility.py` |
| `object_trackers/python` | `srnn_tracker` + `srnn/`, `deepsort_tracker`, `botsort_tracker`, `motr_tracker`, `siammask_tracker`, `mdnet_tracker`, `sam3_tracker`, shared `kalman.py` |
| `classifiers/python` | `netharn_classifier`, `netharn_refiner` |
| `segmentation/python` | `sam2_segmenter`, `sam3_segmenter`, `sam3_text_query`, `sam2_refiner`, `sam3_refiner`, `sam3_utilities` |
| `descriptors/python` | `torchvision_descriptors`, `torchvision_augment_process`, `torchvision/*_feature_extractor`, `resnet_augmenter` |
| `measurement/python` | `foundation_stereo`, `fast_foundation_stereo`, `dino_matcher`, `pair_stereo_tracks.py` (merged), `minima_loftr/` |
| `training/python` | `kwcoco_train_detector`, all `*_trainer.py` and `*_launcher.py`, `netharn/`, `siammask_trainer`, `cutler_trainer` |
| `training/python/export` | `convert_to_onnx_process.py`, `onnx_exporters/*` |
| wheels (P9) | `learn/{pydensecrf,tokencut,cutler,panopticapi,mmdet}`, `remax/`, `siammask/` library part, `mdnet/` |
| deleted | `remax_base_trainer.py`, `remax_detector_example.py`, `remax_trainer_example.py`, `detectron2/` |

### Remaining plugins

| Plugin | Destination |
|---|---|
| `onnx` | `object_detectors/python` (`onnx_predictor`, `onnx_detector`); `classifiers/python` (`onnx_clf_predictor`, `onnx_classifier`, `onnx_refiner`); `measurement/python` (`epipolar_matcher`, `epipolar_dino_matcher`, `fast_foundation_stereo`, `triangulate`, `geometry_numpy`, `calibration_io`, `export_stereo_mapping`, `run_epipolar_onnx`); `README.md` -> `docs/manual` |
| `darknet` | `object_detectors/` (`darknet_detector`, `darknet_custom_resize`), `training/` (`darknet_trainer`); gated on `VIAME_ENABLE_DARKNET` |
| `svm` | `classifiers/` (`refine_detections_svm`), `training/` (`train_detector_svm`, `train_svm_models_process`), `descriptors/` (`iqr_session_svm.h`, `process_query_process`) |
| `cppdb` | `file_io/database/` (`*_db` algos), `descriptors/` (`*_db_process`, merged with CSV variants) |
| `vxl` | replaced in P3 (`image_processing/` on `image_ops`); `format_images_srm_process` -> `descriptors/` |
| `colmap` | `measurement/python` (`reconstruction`, `prior_coverage_sfm`), `image_processing/python` (`colmap_registration`) |
| `seagis` | `measurement/` (`seagis_measurement_process`, mock lib under `measurement/tests/`) |
| `vertex-ai` | `object_detectors/` (`vertex_ai_detector`), `training/` (`vertex_ai_trainer`), `utilities/` (`vertex_ai_client`), python handlers -> `tools/vertex_ai/`; option declared |
| `claude` | `training/` (`train_supervisor`) |
| `examples`, `templates` | `library/examples/` (templates updated to the new helpers) |
| `matlab`, `tensorflow` | dropped (open decision 4 for matlab) |

### `tools/`

Stays `tools/`. `csv.cxx`, `json.cxx` use `third_party/rapidjson` directly.
New applets: `registry-dump`, `pipe-check`. Python scripts still install
to `configs/` for DIVE.

## 4. Dead code removed in P2

- `plugins/pytorch/remax_base_trainer.py`, `remax_detector_example.py`,
  `remax_trainer_example.py`, `plugins/pytorch/detectron2/`
- `plugins/opencv/stereo_comp.py`, `stereo_demo.py`, `stereo_expt.py`,
  `register_algorithms.h`, `stereo_processes.py::MeasureProcess`
- `plugins/matlab/{camtrawl,annosaurus}` and `.mexmaci64` binaries
- `packages/itk-modules/trimmed-point-set`,
  `packages/pytorch-libs/{mmdet-to-tensorrt,fast-foundation-stereo}`
- Stale pipeline references `pytorch_augmentation`, `mdnet_tracker`
  (process names that resolve to nothing today) recorded in `removed.json`
