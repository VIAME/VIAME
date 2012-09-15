#!/bin/sh

git="$( which git )"

die () {
  echo >&2 "$@"
  exit 1
}

[ -x "$git" ] || \
  die "Could not find 'git' in \$PATH ($PATH)."

this="$0"
this_dir="$( dirname "$this" )"

cd "$this_dir"

"$git" rev-parse --is-inside-work-tree 2> /dev/null || \
  die "Does not seem to be a git repository."

git_dir="$( git rev-parse --git-dir )"
hook_dir="$git_dir/hooks"

cd "$hook_dir"

"$git" init && \
  "$git" remote add origin ../..
