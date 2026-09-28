# Compatibility baseline

`registry.json`, `pipes.json` and `install.txt` record what a build
registers, what every shipped pipeline resolves to, and what the build
installs where. They are the contract the `lite` branch work is
held to: names may be added, renamed behind an alias, or removed on purpose
through `removed.json`, but nothing may disappear or change its defaults by
accident.

## Recording a change

**Do not regenerate `registry.json` wholesale.** It is not a snapshot of the
current build; it is `main`'s surface, and `removed.json` carries a reason for
each of the 157 names this branch no longer registers. A fresh dump drops all
157 and the reasons stop describing anything -- the contract's whole content
is the difference between the two files.

How to record a change depends on what changed:

| what changed | where it goes |
|---|---|
| a whole implementation, gone on purpose | `removed.json`, `{kind, interface, name, phase, reason}` |
| a name, config key or port gone until a later phase brings it back | `pending.json`, same shape plus `config_keys` / `ports` / `whole_name` |
| a config key gone for good, or a **default changed** | that entry in `registry.json`, and only that entry -- there is no tolerance lane for a changed default |
| a pipeline file gone | `removed_pipes.json` |
| a pipeline's status, or what a process resolves to | that entry in `pipes.json` |
| what the build installs | re-record `install.txt`; it is a flat manifest with no exceptions list |

To update single entries, dump beside the baseline and copy across only the
entries you mean to move:

```
source <install>/setup_viame.sh
viame registry-dump --json --introspect --output /tmp/registry.json
viame pipe-check   --all --json --output /tmp/pipes.json
```

`--introspect` is not optional. It imports each python implementation so its
defaults can be read, and it is what `baseline:registry` runs; a dump without
it records an error for every python algorithm and the comparison then skips
those entries.

The install manifest is the one that is re-recorded whole:

```
cmake --install <build> > install.log
python3 tests/baseline/install_manifest.py install.log \
        --prefix <install> --record tests/baseline/install.txt
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
| `install.txt` | Every file VIAME's own build installs, taken from what the install step says it placed |
| `critical.txt` | The `CRITICAL`-labelled ctest names at the time the baseline was taken |

## Known gaps

- 58 python-registered algorithms report an `error` instead of their config
  keys: the pybind trampoline returns `config_block` by copy and the type is
  non-copyable. Their names are still recorded, so their disappearance is
  caught; their defaults are not.
- `install.txt` is built from the install step's own output rather than by
  listing the tree. The first version of it listed the tree and was wrong:
  VIAME installs into a prefix it shares with fletch, so 4,439 of the 5,584
  paths it recorded were fletch's headers -- `include/cppdb`, GDAL's -- and
  `make install` only ever adds, so the tree also held what older builds had
  left behind, including a whole `lib/cmake/kwiver` that nothing has
  installed since P5-T05. Reading the tree answers "what has accumulated
  here", which is not the question. The check re-runs the install, which
  takes about two seconds when everything is up to date.
- The baseline is from a CUDA build. A CPU-only build registers a subset, so
  comparing a CPU dump against this baseline reports the GPU-only names as
  gone.
