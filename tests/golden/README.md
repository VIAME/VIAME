# Golden behaviour recordings

A recording is what an implementation does *before* it is replaced. It is the
evidence that a replacement behaves the same, which registry and pipeline
checks cannot give: those say a name still resolves, not that it still
computes the same pixels.

Each group under this directory belongs to one dependency removal:

| Group | Recorded from | Replaced by |
|---|---|---|
| `vxl` | `arrows/vxl` filters and image_io, `plugins/vxl` filters | phase 3 |

## Layout

```
inputs/                     committed fixture images, never regenerated
<group>/manifest.json       every case: impl, config, inputs, output digests
<group>/<kind>/<impl>/<variant>/<input>.png|.npz
```

Outputs are PNG when the result is 8 or 16 bit (viewable, exact) and a
compressed `.npz` otherwise, which covers float and boolean results and
multi-channel float. Neither ever rescales.

The fixtures are small synthetic images built by `fixtures.py` from a fixed
seed: a gradient, bars and a disc so conversion and morphology have edges, a
mask for the filters that only take one, seeded noise so the histogram and
percentile paths see a distribution, and a 12 frame sequence with a moving
disc for the temporal filters. They are committed, so replaying a golden does
not depend on `fixtures.py` or on any image decoder.

## Recording

Against an install that still has the old implementation:

```
source <install>/setup_viame.sh
python3 tests/golden/record.py vxl
```

`record.py` refuses to overwrite an existing recording without `--force`, so
a golden cannot be quietly redefined by the code it is meant to be checking.
Re-record only when a task says the recorded behaviour changes, and say why in
`design/STATUS.md`.

The config variants in `cases.py` are the ones the shipped pipelines use, read
out of every `.pipe` and `.conf` in the install, plus each implementation's own
defaults. Adding a variant means adding it there and re-recording.

## Replaying

```
ctest -L GOLDEN
```

or directly:

```
source <install>/setup_viame.sh
python3 -m pytest tests/golden/test_golden.py -v
```

Shape and dtype always have to match. Values have to match within the
tolerance for that implementation in `test_golden.py`, which is exact by
default; a replacement that genuinely cannot be bit exact adds its own entry
there with the reason and the measured numbers.

## Cases whose values are not a contract

`cases.py` lists five, with reasons. Two are `vxl_convert_image`'s
`random_grayscale` augmentation, which draws from the global RNG. The other
three are VXL bugs found while recording:

- `vxl_threshold` in percentile mode on a multi-plane image returns an
  uninitialised buffer. `percentile_threshold_above` sizes its output to one
  plane, then hands `vil_plane( dst, 0 )` to `vil_threshold_above`, which
  resizes the view it is given to the source's plane count; the resize
  detaches the view and `dst` is never written. Single-plane inputs are fine,
  and the one shipped pipeline that uses `vxl_threshold` uses absolute mode,
  so the path is latent.
- `vxl_color_commonality` in grid mode leaves part of its output
  uninitialised, and differs from call to call inside one process.

For these the recording pins the shape, dtype and that the configuration is
accepted, and nothing else. The replacement should be correct rather than
bug compatible.
