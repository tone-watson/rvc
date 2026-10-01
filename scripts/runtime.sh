#!/bin/sh
# Farm Linux server runtime; the application's version remains in farm.json.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"
uv_bin=${UV_BIN:-/home/gradywoodruff/.local/bin/uv}
export UV_CACHE_DIR=/srv/farm/.uv/cache
export UV_PYTHON_INSTALL_DIR=/srv/farm/.uv/python
export UV_PROJECT_ENVIRONMENT="$project_dir/.venv"
# Keep source builds bounded on the shared server.
export UV_CONCURRENT_BUILDS=1
export MAX_JOBS="${MAX_JOBS:-2}"
case "${1:-sync}" in
  sync)
    "$uv_bin" python install 3.11.14 --no-bin \
      --python-downloads-json-url "$project_dir/scripts/python-downloads.json"
    exec "$uv_bin" sync --locked --python 3.11.14 --managed-python --no-python-downloads
    ;;
  check)
    # Offline release check only; does not mutate the running environment.
    exec "$uv_bin" lock --check --offline --python 3.11.14 \
      --managed-python --no-python-downloads
    ;;
  *) echo "Usage: $0 [sync|check]" >&2; exit 2 ;;
esac
