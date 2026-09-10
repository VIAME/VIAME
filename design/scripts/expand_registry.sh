#!/bin/bash
# Preprocess every register_algorithms.cxx with the include paths its own
# target uses, so that the registration macros expand and the factory names
# become literal. Defines are dropped: the ones that matter here come from
# generated export headers, and the rest carry quotes the shell would have to
# re-escape.
#   design/scripts/expand_registry.sh <build dir> <output dir>
B=${1:?usage: expand_registry.sh <build dir> <output dir>}
S=$(cd "$(dirname "$0")/../.." && pwd)
out=${2:?usage: expand_registry.sh <build dir> <output dir>}
mkdir -p "$out"
: > "$out/errors.txt"
for src in $S/library/*/register_algorithms.cxx $S/plugins/*/register_algorithms.cxx; do
  dir=$(basename $(dirname $src))
  parent=$(basename $(dirname $(dirname $src)))
  flags=$(ls $B/$parent/$dir/CMakeFiles/*plugin.dir/flags.make 2>/dev/null | head -1)
  [ -z "$flags" ] && { echo "no flags for $src" >> "$out/errors.txt"; continue; }
  inc=$(grep '^CXX_INCLUDES' "$flags" | head -1 | sed 's/^CXX_INCLUDES = //')
  g++ -E -std=c++17 $inc "$src" -o "$out/$parent-$dir.i" 2>>"$out/errors.txt" \
    || echo "FAILED $src" >> "$out/errors.txt"
done
