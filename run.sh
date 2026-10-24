#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p build output

# Configure and build (Release by default)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j

# Try common executable names
EXE=""
for cand in build/opt-flow-parallel build/Release/opt-flow-parallel.exe build/Release/opt-flow-parallel; do
  if [[ -x "$cand" ]]; then
    EXE="$cand"
    break
  fi
done

if [[ -z "$EXE" ]]; then
  echo "Error: built executable not found in ./build."
  echo "Look for the target name in CMakeLists.txt."
  exit 1
fi

# Run with sample inputs
"$EXE" ./data/rm1.jpg ./data/rm2.jpg