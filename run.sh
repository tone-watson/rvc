#!/bin/sh
# Farm server launcher. Install the locked runtime separately before launch.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"
rvc_python=${FARM_RVC_PYTHON:-$project_dir/.venv/bin/python}
if [ ! -x "$rvc_python" ]; then
  echo "RVC Python is unavailable: $rvc_python. Run sh scripts/runtime.sh sync." >&2
  exit 1
fi
# Config defaults child preprocess/extract/train commands to sys.executable.
exec "$rvc_python" infer-web.py --port 7861 --noautoopen "$@"
