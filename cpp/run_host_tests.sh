#!/usr/bin/env bash
# Host tests for the Rust core and its C ABI.
#
# Runs everything that can run without an Android device: the Rust unit tests,
# the lints, and the C++ smoke test that links the core's `.so` and calls the
# `gbr_*` symbols from real C++. The last one is the important one - it is what
# proves the `extern "C"` boundary and the `#[repr(C)]` layouts, which a Rust-only
# test cannot.
#
# Usage: bash cpp/run_host_tests.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "== rustfmt =="
cargo fmt --all -- --check

echo "== clippy =="
cargo clippy --all-targets -- -D warnings

echo "== rust unit tests =="
cargo test --all-features

echo "== release cdylib =="
cargo build --release

echo "== C++ FFI smoke test =="
# g++ or clang++ both work; the mod itself is built by the NDK clang.
CXX="${CXX:-g++}"
OUT="$(mktemp -d)/ffi_smoke"
"${CXX}" -std=c++20 -Wall -Wextra -Werror \
    -Icpp cpp/ffi_smoke_test.cpp \
    -Ltarget/release -lglowberry_rust -Wl,-rpath,"${ROOT}/target/release" \
    -o "${OUT}"
"${OUT}"

echo
echo "all host tests passed"
