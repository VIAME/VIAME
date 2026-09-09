# Phase 4: drop FFmpeg

Goal: video read/write is python on PyAV; `arrows/ffmpeg` not built.
References: lite-removals.md §3. Requires open decision 1 = yes (python
wheels allowed); if not, mark P4-T02 blocked.

### P4-T01 Record video golden data and benchmark baseline
Depends: P3-T08
Do:
- From the current install, for each test video in `pipelines_test_data` (add a 10-bit and a variable-frame-rate sample if missing): dump frame count, per-frame timestamps (`viame` small applet or a python script via `kwiver.vital.algo.VideoInput`), md5 of frames 1, N/2, N, and decode throughput (frames/s) for 1080p h264. Commit under `tests/golden/video/`.
Done when:
- Manifest committed with numbers; throughput recorded in STATUS.md.

### P4-T02 PyAV `video_input`
Depends: P4-T01
Do:
- `library/video_io/python/pyav_video_input.py` implementing `kwiver.vital.algo.VideoInput`; config keys and semantics per lite-removals.md §3.1 (copy keys from `registry.json` entry for `video_input/ffmpeg`). Register `ffmpeg` (python impl replaces C++ one; the C++ registration is removed in P4-T05) plus aliases `vidl_ffmpeg`, `pyav`. Zero-copy numpy -> `ImageContainer`.
- Add `av` to `base.in`/lock. Unit tests: frame count, timestamps within 1 us of golden, md5 match for the three frames, seek to frame k then next_frame equals frame k+1.
Done when:
- Tests pass; throughput >= 60 % of the C++ baseline recorded in STATUS.md (target >= 150 fps 1080p on the reference machine; if below, implement P4-T04 before continuing).

### P4-T03 PyAV `video_output`
Depends: P4-T02
Do:
- `pyav_video_output.py` implementing `VideoOutput` with keys per §3.2; flush/close in finalize. Test: write 30 frames, read back with P4-T02, frame count and mean abs pixel diff < 3 (lossy codec).
Done when:
- Test passes; `filter_to_video.pipe` golden (frame count + duration) passes.

### P4-T04 `ffmpeg_cli` fallback reader
Depends: P4-T02
Do:
- Subprocess rawvideo reader used when `import av` fails or when `use_cli=true`; binary from `imageio-ffmpeg` (add to lock). Same tests as P4-T02 minus seek precision (documented).
Done when:
- Tests pass with `av` uninstalled in a venv.

### P4-T05 Switch FFmpeg off
Depends: P4-T03, P4-T04
Do:
- `KWIVER_ENABLE_FFMPEG=OFF`; delete `VIAME_ENABLE_FFMPEG*` options, `FindFFMPEG.cmake`, the ffmpeg lines in `viame_dependencies.cmake`; C++ `ffmpeg` registrations gone, python ones own the names. `ldd` of plugins shows no `libav*`.
- DIVE smoke: open a video dataset, run detector, export video.
Done when:
- Build from clean without FFmpeg dev packages installed; BASELINE, CRITICAL, GOLDEN pass; smoke passes.
