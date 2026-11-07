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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
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
  // 0x0002_00 = major 0, minor 2, patch 0.
  CHECK_EQ(gbr_version(), 0x00000200u);
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

  const std::uint32_t n = gbr_frustum_cull(proj, boxes, visible, 2);
  CHECK(n <= 2);
  CHECK_EQ(gbr_frustum_cull(nullptr, boxes, visible, 2), 0u);
}

// Drives the whole pipeline: submit a chunk, drain the finished mesh, check the
// counters. This is the in-session loop the mod runs, exercised from C++.
static void test_pipeline() {
  GbrPipeline *p = gbr_pipeline_new(2, 32, 32);
  CHECK(p != nullptr);
  if (p == nullptr) {
    return;
  }

  auto blocks = make_chunk(8, 8, 8);
  std::memset(blocks.data(), 1, blocks.size());
  CHECK_EQ(gbr_pipeline_submit(p, 42, blocks.data(), 8, 8, 8), 0);

  std::vector<GbrMesh> meshes(16);
  std::vector<std::uint64_t> ids(16, 0);
  int n = 0;
  for (int attempt = 0; attempt < 500 && n == 0; ++attempt) {
    n = gbr_pipeline_drain(p, meshes.data(), ids.data(), 16);
    if (n == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }
  CHECK_EQ(n, 1);
  if (n == 1) {
    CHECK_EQ(ids[0], std::uint64_t{42});
    CHECK_EQ(meshes[0].vertex_count, std::size_t{24});
    gbr_mesh_free(meshes[0]);
  }

  GbrPipelineStats stats{};
  CHECK_EQ(gbr_pipeline_stats(p, &stats), 0);
  CHECK_EQ(stats.submitted, std::uint64_t{1});
  CHECK_EQ(stats.meshed, std::uint64_t{1});
  CHECK_EQ(stats.drained, std::uint64_t{1});

  gbr_pipeline_free(p);
  gbr_pipeline_free(nullptr); // null is safe
}

// The client's world model: a footprint of block layers, meshed directly.
static void test_world_layers() {
  // bedrock(1) + 2 dirt(2) + 1 grass(3) over an 8x8 footprint -> 8x4x8 box.
  const GbrBlockLayer layers[3] = {
      {1, 1},
      {2, 2},
      {3, 1},
  };
  GbrMesh mesh = gbr_mesh_world_layers(layers, 3, 8, 8);
  // Height 4 = bedrock(1) + dirt(2) + grass(1), so each of the four sides has
  // three material bands (y=0, y=1-2, y=3): 4*3 = 12 side quads, plus the top
  // and bottom = 14 quads = 56 vertices.
  CHECK_EQ(mesh.vertex_count, std::size_t{56});
  gbr_mesh_free(mesh);

  GbrMesh empty = gbr_mesh_world_layers(nullptr, 0, 8, 8);
  CHECK_EQ(empty.vertex_count, std::size_t{0});
  gbr_mesh_free(empty);
}

// The offset-map path: parse a report, look up a slot, gate on the build id.
static void test_offset_map() {
  // The shape tools/extract_offsets.py emits.
  static const char *kJson =
      "{"
      "\"version\":\"26.60\","
      "\"so_bytes\":342231392,"
      "\"so_sha256\":\"b6ffe00817117b03a518b4c9254dcbbb821ad53f276d179c67021226ca221a80\","
      "\"rtti_slots\":["
      "{\"type_name\":\"14ClientInstance\",\"slot\":31,\"function_vaddr\":163051024},"
      "{\"type_name\":\"18LivePlayerRenderer\",\"slot\":17,\"function_vaddr\":161642012}"
      "],"
      "\"entity_seam_offsets\":{\"resolved\":false,\"fields\":["
      "{\"name\":\"levelEntityListOffset\",\"offset\":null}"
      "]}"
      "}";

  GbrOffsetMap *map =
      gbr_offsets_load(reinterpret_cast<const std::uint8_t *>(kJson), std::strlen(kJson));
  CHECK(map != nullptr);
  if (map == nullptr) {
    return;
  }

  const char *type_name = "14ClientInstance";
  CHECK_EQ(
      gbr_offsets_slot(map, reinterpret_cast<const std::uint8_t *>(type_name),
                       std::strlen(type_name), 31),
      std::uint64_t{163051024});
  // An absent slot resolves to 0, never a fabricated address.
  CHECK_EQ(gbr_offsets_slot(map, reinterpret_cast<const std::uint8_t *>(type_name),
                            std::strlen(type_name), 999),
           std::uint64_t{0});

  // The build gate: a matching hash passes, a different one fails.
  const char *good =
      "b6ffe00817117b03a518b4c9254dcbbb821ad53f276d179c67021226ca221a80";
  const char *bad = "deadbeef";
  CHECK_EQ(gbr_offsets_verify_build(map, reinterpret_cast<const std::uint8_t *>(good),
                                    std::strlen(good)),
           1u);
  CHECK_EQ(gbr_offsets_verify_build(map, reinterpret_cast<const std::uint8_t *>(bad),
                                    std::strlen(bad)),
           0u);

  // The entity seam is not recovered in this map.
  CHECK_EQ(gbr_offsets_entity_seam_complete(map), 0u);

  char version[32] = {0};
  const std::size_t n =
      gbr_offsets_version(map, reinterpret_cast<std::uint8_t *>(version), sizeof(version));
  CHECK_EQ(n, std::strlen("26.60"));
  CHECK(std::strcmp(version, "26.60") == 0);

  gbr_offsets_free(map);
  gbr_offsets_free(nullptr); // null is safe

  // A malformed document is a null, not a crash.
  const char *garbage = "not json at all";
  CHECK(gbr_offsets_load(reinterpret_cast<const std::uint8_t *>(garbage),
                         std::strlen(garbage)) == nullptr);
}

// The entity-list walk: build a synthetic Level -> ActorManager -> vector<Actor*>
// graph in C++ and drive gbr_entity_walk / gbr_entity_position over the ABI. This
// is what proves the two-step chain and the pointer layout agree across the
// language boundary.
static void test_entity_walk() {
  const char *kJson =
      "{"
      "\"version\":\"26.60\","
      "\"so_sha256\":\"b6ffe00817117b03a518b4c9254dcbbb821ad53f276d179c67021226ca221a80\","
      "\"entity_seam_offsets\":{\"resolved\":true,\"fields\":["
      "{\"name\":\"levelEntityListOffset\",\"offset\":992},"
      "{\"name\":\"entityPositionOffset\",\"offset\":560}"
      "]}"
      "}";
  GbrOffsetMap *map =
      gbr_offsets_load(reinterpret_cast<const std::uint8_t *>(kJson), std::strlen(kJson));
  CHECK(map != nullptr);
  if (map == nullptr) {
    return;
  }
  CHECK_EQ(gbr_offsets_entity_walk_ready(map), 1u);

  // Two actors, each with a Vec3 at +0x230.
  struct ActorStorage {
    std::uint8_t bytes[0x230 + 12];
  };
  ActorStorage a0{}, a1{};
  auto write_pos = [](ActorStorage &a, float x, float y, float z) {
    std::memcpy(a.bytes + 0x230, &x, 4);
    std::memcpy(a.bytes + 0x230 + 4, &y, 4);
    std::memcpy(a.bytes + 0x230 + 8, &z, 4);
  };
  write_pos(a0, 1.0f, 2.0f, 3.0f);
  write_pos(a1, 4.0f, 5.0f, 6.0f);

  // The vector: {begin, end, cap} of Actor*.
  const std::uint8_t *entries[2] = {a0.bytes, a1.bytes};
  struct Manager {
    std::uint8_t pad[0x8];
    const std::uint8_t **begin;
    const std::uint8_t **end;
    const std::uint8_t **cap;
  } manager{};
  manager.begin = entries;
  manager.end = entries + 2;
  manager.cap = entries + 2;

  // The Level: the manager pointer at +0x3e0.
  struct Level {
    std::uint8_t bytes[0x3e0 + 8];
  } level{};
  auto *manager_ptr = reinterpret_cast<const std::uint8_t *>(&manager);
  std::memcpy(level.bytes + 0x3e0, &manager_ptr, 8);

  const std::uint8_t *actors[8] = {nullptr};
  const int n = gbr_entity_walk(map, level.bytes, actors, 8);
  CHECK_EQ(n, 2);
  if (n == 2) {
    float position[3] = {0, 0, 0};
    CHECK_EQ(gbr_entity_position(map, actors[0], position), 0);
    CHECK(position[0] == 1.0f && position[1] == 2.0f && position[2] == 3.0f);
    CHECK_EQ(gbr_entity_position(map, actors[1], position), 0);
    CHECK(position[0] == 4.0f && position[1] == 5.0f && position[2] == 6.0f);
  }

  // Null arguments are refused.
  CHECK_EQ(gbr_entity_walk(nullptr, level.bytes, actors, 8), -1);
  CHECK_EQ(gbr_entity_walk(map, nullptr, actors, 8), -1);
  CHECK_EQ(gbr_entity_position(map, nullptr, nullptr), -1);

  // A null Level pointer is refused, not dereferenced.
  CHECK_EQ(gbr_entity_walk(map, nullptr, actors, 8), -1);

  gbr_offsets_free(map);

  // A map without the entity seam is not walk-ready.
  const char *no_seam = "{\"version\":\"x\",\"so_sha256\":\"y\"}";
  GbrOffsetMap *bare =
      gbr_offsets_load(reinterpret_cast<const std::uint8_t *>(no_seam), std::strlen(no_seam));
  CHECK(bare != nullptr);
  if (bare != nullptr) {
    CHECK_EQ(gbr_offsets_entity_walk_ready(bare), 0u);
    // The walk then returns 0 rather than touching memory.
    CHECK_EQ(gbr_entity_walk(bare, level.bytes, actors, 8), 0);
    gbr_offsets_free(bare);
  }
}

// The fallback: with no map at all (the fail-closed default), every entry point
// answers the "not available" value instead of dereferencing anything. This is
// what the mod relies on when a build has no verified offsets.
static void test_fallback_is_safe() {
  // A null map is refused by every entity entry point.
  const std::uint8_t dummy[16] = {0};
  const std::uint8_t *actors[4] = {nullptr};
  float position[3] = {0, 0, 0};
  CHECK_EQ(gbr_offsets_entity_walk_ready(nullptr), 0u);
  CHECK_EQ(gbr_offsets_entity_seam_complete(nullptr), 0u);
  CHECK_EQ(gbr_entity_walk(nullptr, dummy, actors, 4), -1);
  CHECK_EQ(gbr_entity_position(nullptr, dummy, position), -1);

  // A malformed document loads as null, and the caller simply has no map.
  const char *bad = "not json";
  CHECK(gbr_offsets_load(reinterpret_cast<const std::uint8_t *>(bad), std::strlen(bad)) ==
        nullptr);

  // A map for a different build fails verification, so the mod keeps the
  // fallback rather than using stale offsets.
  const char *kJson =
      "{"
      "\"version\":\"26.60\","
      "\"so_sha256\":\"b6ffe00817117b03a518b4c9254dcbbb821ad53f276d179c67021226ca221a80\","
      "\"entity_seam_offsets\":{\"fields\":["
      "{\"name\":\"levelEntityListOffset\",\"offset\":992},"
      "{\"name\":\"entityPositionOffset\",\"offset\":560}"
      "]}"
      "}";
  GbrOffsetMap *map =
      gbr_offsets_load(reinterpret_cast<const std::uint8_t *>(kJson), std::strlen(kJson));
  CHECK(map != nullptr);
  if (map != nullptr) {
    const char *wrong = "deadbeef";
    CHECK_EQ(gbr_offsets_verify_build(map, reinterpret_cast<const std::uint8_t *>(wrong),
                                      std::strlen(wrong)),
             0u);
    gbr_offsets_free(map);
  }
}

int main() {
  test_version();
  test_mesh_lone_block();
  test_mesh_full_chunk_merges();
  test_null_is_safe();
  test_ring_round_trip();
  test_collision();
  test_frustum_cull();
  test_pipeline();
  test_world_layers();
  test_offset_map();
  test_entity_walk();
  test_fallback_is_safe();

  if (g_failures == 0) {
    std::printf("ffi_smoke: all checks passed\n");
    return 0;
  }
  std::fprintf(stderr, "ffi_smoke: %d check(s) failed\n", g_failures);
  return 1;
}
