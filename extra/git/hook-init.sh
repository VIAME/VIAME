#!/bin/sh

git="$( which git )"

[ -x "$git" ] || \
  { echo "Could not find 'git' in \$PATH ($PATH)"; exit 1 }

this="$0"
this_dir="$( dirname "$this" )"

cd "$this_dir"

"$git" rev-parse --is-inside-work-tree 2> /dev/null || \
  { echo "Does not seem to be a git repository"; exit 1 }

git_dir="$( git rev-parse --git-dir )"
hook_dir="$git_dir/hooks"

cd "$hook_dir"

"$git" init && \
  "$git" remote add origin ../..
