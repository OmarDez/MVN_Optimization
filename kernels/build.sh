#!/usr/bin/env bash
# Build the angular kernel shared library.
#   bash kernels/build.sh
# Produces kernels/libangular.so, loaded by pymvn/angular.py via ctypes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CC="${CC:-cc}"
ARCH="$(uname -m)"

FLAGS=(-O3 -shared -fPIC -std=c11 -Wall -Wextra)

case "$ARCH" in
  aarch64|arm64)
    # -mcpu=native lets the compiler use every feature the host advertises
    # (SVE2 on Neoverse N2, for example). Fall back if it is unsupported.
    if echo 'int main(void){return 0;}' | "$CC" -mcpu=native -x c - -o /dev/null 2>/dev/null; then
      FLAGS+=(-mcpu=native)
    else
      FLAGS+=(-march=armv8-a)
    fi
    echo "[build] aarch64 detected -- NEON path will be compiled in"
    ;;
  *)
    echo "[build] $ARCH detected -- scalar fallback only (expected off-target)"
    ;;
esac

"$CC" "${FLAGS[@]}" "$HERE/angular_neon.c" -o "$HERE/libangular.so"
echo "[build] wrote $HERE/libangular.so"

python3 - <<'PY'
import pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd()))
try:
    from pymvn.angular import neon_available
    print(f"[build] NEON path active: {neon_available()}")
except Exception as e:
    print(f"[build] (could not verify from Python: {e})")
PY
