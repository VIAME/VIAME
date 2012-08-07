#!/bin/sh

die () {
  echo >&2 "$@"
  exit 1
}

source_dir="$PWD"

[ -e "$source_dir/.git" ] || die "Not at the top of the repository."
[ -d "$source_dir/.git" ] || die "Not a valid repository."

# Rudimentary check for vistk itself.
grep -q vistk_check_compiler_flag CMakeLists.txt || die "You do not seem to be in a vistk repository."

prompt () {
  msg="$1"
  shift

  default=
  [ -n "$1" ] && default="$1"

  [ -z "$default" ] &&
    echo -n "$msg" ||
    echo -n "$msg[$default] "
  read input

  [ -z "${variable}" ] && [ -n "$default" ] &&
    input="$default"
}

echo "Initializing gerrit code review..."
gerrit_dir="$source_dir/.git/gerrit"
if [ -e "$gerrit_dir" ]; then
  echo "WARNING: '$gerrit_dir' already exists."

  prompt "Overwrite existing gerrit configuration? [y/N] "
  overwrite="$input"
  if [ "$overwrite" = "y" ] || [ "$overwrite" = "Y" ]; then
    rm -rf "$gerrit_dir"
  else
    die "Aborting gerrit initialization."
  fi
fi

mkdir -p "$gerrit_dir"
cd "$gerrit_dir"
git init
git pull ../.. remotes/origin/gerrit
cd "$source_dir"

git config hooks.GerritId false

gerrit_ssh_test=0
while [ "$gerrit_ssh_test" -ne 1 ]; do

  prompt "Gerrit hostname: " cvreview.kitwarein.com
  gerrit_hostname="$input"

  prompt "Gerrit ssh port: " 29418
  gerrit_port="$input"

  prompt "Gerrit username: " "$( git config user.email | sed -e 's|@.*||' )"
  gerrit_user="$input"

  prompt "Gerrit project: " VisTK
  gerrit_project="$input"

  prompt "Gerrit target branch: " master
  gerrit_target_branch="$input"

  echo "Testing SSH connection..."
  ssh -p "${gerrit_port}" "${gerrit_user}@${gerrit_hostname}"
  if [ $? -eq 127 ]; then
    gerrit_ssh_test=1
    echo "SSH connection successful!"
  else
    echo "SSH connection to the Gerrit server failed"
  fi
done

git config gerrit2.hostname "${gerrit_hostname}"
git config gerrit2.port "${gerrit_port}"
git config gerrit2.username "${gerrit_user}"
git config gerrit2.project "${gerrit_project}"
git config gerrit2.targetbranch "${gerrit_target_branch}"

if git remote | grep -q -e '^gerrit$'; then
  git remote rm gerrit2
fi
git remote add gerrit2 ssh://${gerrit_user}@${gerrit_hostname}:${gerrit_port}/${gerrit_project}.git

git config alias.gerrit2 "!bash ${gerrit_dir}/git-gerrit2-wrapper.sh"
