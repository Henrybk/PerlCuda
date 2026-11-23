#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
g++ -O3 -std=c++17 -fPIC -shared native/blockdist_cpu.cpp -o build/libblockdist_cpu.so
if command -v nvcc >/dev/null 2>&1; then
  nvcc -O3 --shared -Xcompiler -fPIC -o build/libblockdist_cuda.so native/blockdist_cuda.cu
else
  echo 'nvcc not found: built CPU only; CUDA tests will be skipped.'
fi
