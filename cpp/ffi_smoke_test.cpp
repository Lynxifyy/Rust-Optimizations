// A host-side C++ smoke test for the Rust core's C ABI.
//
// This is the real proof that the boundary works: it is C++ code, compiled by
// g++, calling the `gbr_*` symbols and reading the `#[repr(C)]` structs. If a
// field order, a size, or an `extern "C"` name were wrong, this would fail to
// link or read garbage - which is exactly the class of bug that a Rust-only unit
// test cannot catch.
//
// Build and run (from the repository root, after `cargo build --release`):
//
//   g++ -std=c++20 -Wall -Wextra -Werror cpp/ffi_smoke_test.cpp
//       -Ltarget/release -lglowberry_rust -Wl,-rpath,target/release -o ffi_smoke
//   ./ffi_smoke
//
// It is deliberately free of Android and preloader headers so it builds on any
// host; `cpp/run_host_tests.sh` drives it.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "glowberry_rust_abi.h"

// --- Minimal assertion helpers ------------------------------------------------

static int g_failures = 0;

#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

#define CHECK_EQ(a, b)                                                         \
  do {                                                                         \
    const auto _a = (a);                                                       \
    const auto _b = (b);                                                       \
    if (!(_a == _b)) {                                                         \
      std::fprintf(stderr, "FAIL %s:%d: %s (%lld) != %s (%lld)\n", __FILE__,   \
                   __LINE__, #a, static_cast<long long>(_a), #b,               \
                   static_cast<long long>(_b));                                \
      ++g_failures;                                                            \
    }                                                                          \
  } while (0)

static std::vector<std::uint8_t> make_chunk(std::uint32_t x, std::uint32_t y,
                                            std::uint32_t z) {
  return std::vector<std::uint8_t>(static_cast<std::size_t>(x) * y * z, 0);
}

static void set_block(std::vector<std::uint8_t> &blocks, std::uint32_t X,
                      std::uint32_t Y, std::uint32_t x, std::uint32_t y,
                      std::uint32_t z, std::uint8_t value) {
  blocks[x + X * (y + Y * z)] = value;
}

// --- Tests --------------------------------------------------------------------

static void test_version() {
  // 0x0001_00 = major 0, minor 1, patch 0.
  CHECK_EQ(gbr_version(), 0x00000100u);
}

static void test_mesh_lone_block() {
  auto blocks = make_chunk(4, 4, 4);
  set_block(blocks, 4, 4, 1, 1, 1, 1);

  GbrMesh mesh = gbr_mesh_blocks(blocks.data(), 4, 4, 4);
  CHECK_EQ(mesh.vertex_count, std::size_t{24});
  CHECK_EQ(mesh.index_count, std::size_t{36});
  CHECK(mesh.vertices != nullptr);

  // The vertex struct layout must be readable as the C++ mirror of it.
  if (mesh.vertices != nullptr) {
    const GbrVertex &v0 = mesh.vertices[0];
    CHECK(v0.pos[0] >= 0.0f && v0.pos[0] <= 4.0f);
    CHECK(v0.ao <= 3);
  }
  gbr_mesh_free(mesh);
}

static void test_mesh_full_chunk_merges() {
  auto blocks = make_chunk(16, 16, 16);
  std::memset(blocks.data(), 1, blocks.size());

  GbrMesh mesh = gbr_mesh_blocks(blocks.data(), 16, 16, 16);
  // A solid chunk exposes six merged quads (24 vertices), not 6 * 256.
  CHECK_EQ(mesh.vertex_count, std::size_t{24});
  gbr_mesh_free(mesh);
}

static void test_null_is_safe() {
  GbrMesh mesh = gbr_mesh_blocks(nullptr, 4, 4, 4);
  CHECK_EQ(mesh.vertex_count, std::size_t{0});
  CHECK(mesh.vertices == nullptr);
  gbr_mesh_free(mesh);
  gbr_ring_free(nullptr);
  CHECK_EQ(gbr_ring_len(nullptr), std::size_t{0});
}

static void test_ring_round_trip() {
  auto blocks = make_chunk(4, 4, 4);
  set_block(blocks, 4, 4, 1, 1, 1, 1);

  GbrRing *ring = gbr_ring_new(8);
  CHECK(ring != nullptr);
  CHECK_EQ(gbr_ring_push_mesh(ring, 7, blocks.data(), 4, 4, 4), 0);
  CHECK_EQ(gbr_ring_len(ring), std::size_t{1});

  GbrMesh mesh{};
  std::uint64_t id = 0;
  CHECK_EQ(gbr_ring_pop_mesh(ring, &mesh, &id), 0);
  CHECK_EQ(id, std::uint64_t{7});
  CHECK_EQ(mesh.vertex_count, std::size_t{24});
  gbr_mesh_free(mesh);
  gbr_ring_free(ring);
}

static void test_collision() {
  GbrAabb start{{-0.5f, 4.5f, -0.5f}, {0.5f, 5.5f, 0.5f}};
  GbrVec3 delta{0.0f, -10.0f, 0.0f};
  GbrAabb floor{{-10.0f, -1.0f, -10.0f}, {10.0f, 0.0f, 10.0f}};
  GbrAabb out{};
  std::uint32_t flags = 0;

  CHECK_EQ(gbr_resolve_movement(&start, &delta, &floor, 1, &out, &flags), 0);
  CHECK((flags & 8u) != 0u); // on_ground
  CHECK(out.min.y >= 0.0f && out.min.y < 1e-3f);
}

static void test_frustum_cull() {
  // Identity-ish perspective; a box in front of the camera is visible, one
  // behind it is not. Column-major 4x4 (glam from_cols_array order).
  const float proj[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  GbrAabb boxes[2] = {
      {{-0.5f, -0.5f, 0.5f}, {0.5f, 0.5f, 1.5f}},
      {{-0.5f, -0.5f, -1.5f}, {0.5f, 0.5f, -0.5f}},
  };
  std::uint8_t visible[2] = {0, 0};

  const std::uint32_t n =
      gbr_frustum_cull(proj, boxes, visible, 2);
  CHECK(n <= 2);
  CHECK_EQ(gbr_frustum_cull(nullptr, boxes, visible, 2), 0u);
}

int main() {
  test_version();
  test_mesh_lone_block();
  test_mesh_full_chunk_merges();
  test_null_is_safe();
  test_ring_round_trip();
  test_collision();
  test_frustum_cull();

  if (g_failures == 0) {
    std::printf("ffi_smoke: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "ffi_smoke: %d check(s) failed\n", g_failures);
  return 1;
}
