# Compatibility baseline

`registry.json`, `pipes.json` and `install.txt` record what a build
registers, what every shipped pipeline resolves to, and what the build
installs where. They are the contract the `lite` branch work is
held to: names may be added, renamed behind an alias, or removed on purpose
through `removed.json`, but nothing may disappear or change its defaults by
accident.

## Regenerating

Run against an install whose behaviour you want to become the new baseline:

```
source <install>/setup_viame.sh
viame registry-dump --json --output tests/baseline/registry.json
viame pipe-check   --all --json --output tests/baseline/pipes.json
python3 tests/baseline/install_manifest.py <install> \
        --record tests/baseline/install.txt
```

Both dumps are deterministic; running either twice gives identical bytes.
`pipe-check --all` walks `configs/pipelines`, `configs/add-ons` and `examples`
under `$VIAME_INSTALL`, so the add-on model packs the build enabled are part of
the baseline. Regenerate only when a task says the baseline moves, and say in
`design/STATUS.md` why.

## Checking

```
viame registry-dump --json --output /tmp/registry.json
python3 tests/baseline/compare_registry.py tests/baseline/registry.json \
        /tmp/registry.json --removed tests/baseline/removed.json
viame pipe-check --all --json --output /tmp/pipes.json
python3 tests/baseline/compare_pipes.py tests/baseline/pipes.json /tmp/pipes.json
```

`ctest -L BASELINE` runs both against the current install.

## Files

| File | Contents |
|---|---|
| `registry.json` | Every registered algorithm, process, cluster, applet and scheduler with its config keys and defaults |
| `pipes.json` | Per pipeline/config file: whether it bakes, its processes, and the implementation each `:type` key selects |
| `removed.json` | Names removed on purpose: `{kind, interface, name, phase, reason}` |
| `install.txt` | Every file VIAME's own build installs: `lib/*.so*`, `include/`, the four executables, `lib/cmake/{viame,kwiver,sprokit}` and the `viame` and `kwiver` python packages |
| `critical.txt` | The `CRITICAL`-labelled ctest names at the time the baseline was taken |

## Known gaps

- 58 python-registered algorithms report an `error` instead of their config
  keys: the pybind trampoline returns `config_block` by copy and the type is
  non-copyable. Their names are still recorded, so their disappearance is
  caught; their defaults are not.
- `install.txt` deliberately omits `lib/python3.10/site-packages` at large,
  which is 75,000 files of torch and its dependencies: pip placed them, VIAME
  cannot regress them, and they move whenever a lock file does. It also omits
  the pip console scripts in `bin/` for the same reason. The cost is that a
  VIAME file installed into a directory none of the recorded roots covers
  would not be noticed.
- The baseline is from a CUDA build. A CPU-only build registers a subset, so
  comparing a CPU dump against this baseline reports the GPU-only names as
  gone.
