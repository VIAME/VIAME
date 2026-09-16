"""Fixtures for design/scripts/rename_namespaces.py, before it touches the tree."""

import importlib.util
import os
import re
import sys

spec = importlib.util.spec_from_file_location(
    "rn", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "rename_namespaces.py"))
rn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rn)

CASES = [
    (
        "the nested pair collapses, and one closing brace goes with it",
        """#include <viame/core_types/image.h>

namespace kwiver {

namespace vital {

class VITAL_TYPES_EXPORT image
{
  kwiver::vital::config_block_sptr config() const;
};

} // namespace vital

} // namespace kwiver
""",
        """#include <viame/core_types/image.h>

namespace viame {

class VITAL_TYPES_EXPORT image
{
  viame::config_block_sptr config() const;
};

} // namespace viame
""",
    ),
    (
        "a sub-namespace inside the pair keeps its own braces",
        """namespace kwiver {
namespace vital {
namespace algo {

class algorithm;

} // namespace algo
} // end namespace vital
} // end namespace kwiver
""",
        """namespace viame {
namespace algo {

class algorithm;

} // namespace algo
} // namespace viame
""",
    ),
    (
        "the joined form renames in place",
        """namespace kwiver::vital::python {

class algorithm_trampoline;

} // namespace kwiver::vital::python
""",
        """namespace viame::python {

class algorithm_trampoline;

} // namespace viame::python
""",
    ),
    (
        "a joined sub-namespace follows its prefix",
        """namespace kwiver::vital::streamable {

class thing;

} // namespace kwiver::vital::streamable
""",
        """namespace viame::streamable {

class thing;

} // namespace viame::streamable
""",
    ),
    (
        "sprokit becomes a two-level namespace with one brace",
        """namespace sprokit {

class process;
sprokit::process_t make();

} // namespace sprokit
""",
        """namespace viame::pipeline {

class process;
viame::pipeline::process_t make();

} // namespace viame::pipeline
""",
    ),
    (
        "a viame file only has its qualified uses rewritten",
        """namespace viame {
namespace core {

using config_block_sptr = kwiver::vital::config_block_sptr;
sprokit::process_t p;

} // namespace core
} // end namespace viame
""",
        """namespace viame {
namespace core {

using config_block_sptr = viame::config_block_sptr;
viame::pipeline::process_t p;

} // namespace core
} // end namespace viame
""",
    ),
    (
        # adapter_types.h: vital closes on its own line, then adapter and
        # kwiver close together.
        "two namespaces closed on one line",
        """namespace kwiver {
namespace vital {

class datum;

}

namespace adapter{

class port;

} } // end namespace
""",
        """namespace viame {
class datum;

namespace adapter {

class port;

} } // namespace viame
""",
    ),
    (
        "processes declared straight in namespace kwiver follow it too",
        """namespace kwiver {

class refine_detections_process;

}

void reg()
{
  add< kwiver::refine_detections_process >( "refine_detections" );
}
""",
        """namespace viame {

class refine_detections_process;

}

void reg()
{
  add< viame::refine_detections_process >( "refine_detections" );
}
""",
    ),
    (
        "braces in strings and comments are not braces",
        """namespace sprokit {

// a brace in a comment: {
const char* brace = "{";
char c = '}';

} // namespace sprokit
""",
        """namespace viame::pipeline {

// a brace in a comment: {
const char* brace = "{";
char c = '}';

} // namespace viame::pipeline
""",
    ),
    (
        # test_camera_rig_json.cxx: JSON in a raw string, full of braces.
        "braces in a raw string are not braces",
        """namespace sprokit {

const char* json = R"json({ "a": { "b": 1 } })json";

} // namespace sprokit
""",
        """namespace viame::pipeline {

const char* json = R"json({ "a": { "b": 1 } })json";

} // namespace viame::pipeline
""",
    ),
    (
        # datum.cxx and the rest of the bindings: sprokit nested under kwiver,
        # with the binding helpers under that again.
        "the bindings' kwiver::sprokit::python",
        """namespace kwiver {

namespace sprokit {

namespace python {

void bind(sprokit::datum const& d);

} // namespace python

} // namespace sprokit

} // namespace kwiver
""",
        """namespace viame {

namespace pipeline {

namespace python {

void bind(viame::pipeline::datum const& d);

} // namespace python

} // namespace pipeline

} // namespace viame
""",
    ),
    (
        # kwiver_applet.h: tools and vital as siblings, and a separate viame
        # block after them.
        "sibling namespaces, one of which collapses",
        """namespace kwiver {
namespace tools {

class applet;

} // namespace tools

namespace vital {

class config_block;

} // namespace vital
} // namespace kwiver

namespace viame {
class thing;
} // namespace viame
""",
        """namespace viame {
namespace tools {

class applet;

} // namespace tools

class config_block;
} // namespace viame

namespace viame {
class thing;
} // namespace viame
""",
    ),
]

RESIDUE = (
    "two openings on one line are left for a person",
    """namespace kwiver { namespace vital {
class image;
} }
""",
)


def main():
    failures = 0

    for name, before, expected in CASES:
        try:
            got, changed, dropped = rn.rewrite(before)
        except rn.Unfollowable as exc:
            failures += 1
            print("FAIL: {}\n  refused: {}".format(name, exc))
            continue
        if got != expected:
            failures += 1
            print("FAIL: {}\n--- got ---\n{}\n--- expected ---\n{}".format(
                name, got, expected))
        else:
            print("ok: {} ({} namespaces, {} braces dropped)".format(
                name, changed, dropped))

    name, text = RESIDUE
    try:
        got = rn.rewrite(text)[0]
        reported = bool(rn.OLD_NAME.search(got))
        why = "still names kwiver, so the run reports it"
    except rn.Unfollowable as exc:
        reported = True
        why = "refused: {}".format(exc)
    if reported:
        print("ok: {} ({})".format(name, why))
    else:
        failures += 1
        print("FAIL: {}\n{}".format(name, got))

    print("\n{} failure(s)".format(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
