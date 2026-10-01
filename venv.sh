#!/bin/sh
# Compatibility entry point for the single project-local uv environment.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec sh "$project_dir/scripts/runtime.sh" "$@"
