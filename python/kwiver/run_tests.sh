#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nosetests \
  --cover-erase --with-coverage --cover-package=vital \
  "$@" ${SCRIPT_DIR}/vital/tests/
