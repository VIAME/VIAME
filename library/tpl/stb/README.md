# stb

Single-header public-domain image codecs, vendored rather than found: they
are two files, they have no build system, and pinning them here is what makes
a build reproducible without adding a dependency to find.

| File | Version | Purpose |
|---|---|---|
| `stb_image.h` | v2.30 | PNG, JPEG, BMP, GIF, PSD, TGA, HDR, PIC, PNM decode |
| `stb_image_write.h` | v1.16 | PNG, JPEG, BMP, TGA, HDR encode |

Taken from `https://github.com/nothings/stb` at commit
`2c980bb59875b0d32144a71867fbdebb2f77cd20` (2026-08-02).

```
sha256  594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3  stb_image.h
sha256  cbd5f0ad7a9cf4468affb36354a1d2338034f2c12473cf1a8e32053cb6914a05  stb_image_write.h
sha256  bebfe904b14301657e4e5d655c811d51fd31b97c455b9cc2d8600d6bac6cff63  LICENSE
```

Dual licensed MIT / public domain; see `LICENSE`.

## What they do not cover

TIFF. stb has never read or written it, and VIAME's data does: the
calibration and survey imagery is 8 and 16 bit stripped TIFF, uncompressed,
LZW and PackBits. `library/video_io/codecs/tiff.{h,cxx}` is the in-house
reader and writer for that, and anything neither handles falls back to the
python `pil` image_io. `design/lite-removals.md` 2.2 has the shape of the
split.

## Updating

Fetch both headers and `LICENSE` at one commit, replace all three, update the
version numbers, the commit and the digests above, and re-run
`ctest -R golden:replay` -- `tests/golden/codecs` is 18 containers recorded
from OpenCV and is what says a decoder change is visible.
