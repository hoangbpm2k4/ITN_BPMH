#!/usr/bin/env bash
# Chạy toàn bộ test V2. Phải chạy từ thư mục gốc của kho mã.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-/home/hoangbpm/Intekcom/.venv_nemo/bin/python}
exec "$PY" -m unittest discover -s itn_v2/tests -t . "$@"
