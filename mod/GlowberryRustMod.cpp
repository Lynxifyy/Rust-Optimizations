// GlowberryClient native mod: the Rust acceleration core.
//
// This is a preloader mod, not an APK component. The preloader `dlopen`s the
// built `.so` and calls the lifecycle below; the `.so` exports
// `PLGetModRegistration` (emitted by `PL_REGISTER_MOD`) and, because it links
// the Rust static library, the whole `gbr_*` C ABI as well.
//
// What it does in-session:
//   1. Resolves the version of the linked Rust core and logs it, so a session's
//      log line proves which core actually loaded.
//   2. Starts the meshing pipeline: a bounded pool of Rust worker threads that
//      mesh chunks and hand the results back through a lock-free ring.
//   3. Runs a real workload through that pipeline - the client's own world model
//      (a footprint of block layers) split into chunks, every chunk submitted,
//      every finished mesh drained - and logs the counters and the wall-clock
//      time. That is the whole submit -> mesh -> drain loop doing work inside
//      the game process, and the log line is the proof.
//   4. Keeps the pipeline alive and reachable (`g_pipeline`) so the client's own
//      voxel work can submit to it each frame; it is torn down on unload.
//
// What it deliberately does NOT do:
//   - It does not install a hook. The game's chunk mesher is not reachable
//     (libminecraftpe.so is stripped; see the client's
//     `docs/agents/native-entity-camera-feed-what-is-and-is-not-reachable.md`),
//     so a hook here would be guessing an address. The core is a library the
//     client's own voxel work calls, and this mod is the proof it runs.
//   - It does not touch malloc/free. Allocator interposition was removed from the
//     preloader for concrete reasons (hidden visibility meant the definitions
//     never reached the game's allocations; thread_local recurses under the
//     emulated TLS of minSdk 28; a libc block freed by the interposer faults) and
//     must not be reintroduced. See the preloader's AGENTS.md.
//
// Build: `mod/CMakeLists.txt`, cross-compiled for arm64-v8a by the
// `Build .so Mod` workflow.

#include <chrono>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <pl/Config.hpp>
#include <pl/Logger.hpp>
#include <pl/Mod.hpp>

#include "glowberry_rust_abi.h"

namespace {

/// The mod's config. Every field is reflected into `config.json` and
/// `config.schema.json` through the `Schema` specialisation below.
struct GlowberryRustConfig {
  int version = 1;
  bool enabled = true;
  /// Worker threads for the meshing pipeline.
  int pipeline_workers = 2;
  /// Chunk footprint (x/z) in blocks.
  int chunk_xz = 16;
  /// Chunk height (y) in blocks.
  int chunk_y = 16;
  /// World footprint (x/z) in blocks for the startup workload.
  int world_size = 64;
  /// Run the startup workload through the pipeline.
  bool run_workload = true;
};

/// The default flat world, in the same shape `FlatWorldGenerator` builds:
/// one bedrock, two dirt, one grass.
constexpr GbrBlockLayer kDefaultLayers[] = {
    {1, 1}, // minecraft:bedrock
    {2, 2}, // minecraft:dirt
    {3, 1}, // minecraft:grass_block
};
constexpr std::uint32_t kDefaultLayerCount = 3;

/// The live pipeline, reachable by other native code once `load` has run.
GbrPipeline *g_pipeline = nullptr;

/// Splits a `world x height x world` block volume into chunk-sized blocks,
/// clamped at the edges, writing the chunk into `out` (x-fastest).
void extract_chunk(const std::vector<std::uint8_t> &world_blocks, std::uint32_t world,
                   std::uint32_t height, std::uint32_t ox, std::uint32_t oy,
                   std::uint32_t oz, std::uint32_t cw, std::uint32_t ch, std::uint32_t cd,
                   std::vector<std::uint8_t> &out) {
  for (std::uint32_t y = 0; y < ch; ++y) {
    for (std::uint32_t z = 0; z < cd; ++z) {
      for (std::uint32_t x = 0; x < cw; ++x) {
        const std::size_t src = (ox + x) + static_cast<std::size_t>(world) *
                                               ((oy + y) + static_cast<std::size_t>(height) * (oz + z));
        const std::size_t dst =
            x + static_cast<std::size_t>(cw) * (y + static_cast<std::size_t>(ch) * z);
        out[dst] = world_blocks[src];
      }
    }
  }
}

/// Runs the startup workload: split a layered world into chunks, submit every
/// chunk to the pipeline, drain every finished mesh, and report what happened.
///
/// Returns true when every submitted chunk came back, which is the real
/// assertion: the pipeline neither lost nor duplicated work.
bool run_workload(const GlowberryRustConfig &cfg, pl::log::Logger &log) {
  const std::uint32_t world = static_cast<std::uint32_t>(cfg.world_size);
  const std::uint32_t chunk_xz = static_cast<std::uint32_t>(cfg.chunk_xz);
  const std::uint32_t chunk_y = static_cast<std::uint32_t>(cfg.chunk_y);

  // Build the world's block stack once: height = sum of the layer counts.
  std::uint32_t height = 0;
  for (std::uint32_t i = 0; i < kDefaultLayerCount; ++i) {
    height += kDefaultLayers[i].count;
  }
  if (world == 0 || height == 0 || chunk_xz == 0 || chunk_y == 0) {
    log.warn("workload skipped: a dimension is zero");
    return true;
  }

  // Pack the world (x-fastest), then split it into chunks on the fly.
  std::vector<std::uint8_t> blocks(static_cast<std::size_t>(world) * height * world, 0);
  {
    std::uint32_t y = 0;
    for (std::uint32_t i = 0; i < kDefaultLayerCount; ++i) {
      for (std::uint32_t c = 0; c < kDefaultLayers[i].count && y < height; ++c, ++y) {
        for (std::uint32_t x = 0; x < world; ++x) {
          for (std::uint32_t z = 0; z < world; ++z) {
            blocks[x + static_cast<std::size_t>(world) *
                           (y + static_cast<std::size_t>(height) * z)] = kDefaultLayers[i].block;
          }
        }
      }
    }
  }

  const std::uint32_t chunks_x = (world + chunk_xz - 1) / chunk_xz;
  const std::uint32_t chunks_y = (height + chunk_y - 1) / chunk_y;

  const auto started = std::chrono::steady_clock::now();
  std::uint64_t submitted = 0;
  std::uint64_t drained = 0;
  std::uint64_t quads = 0;
  std::vector<std::uint8_t> chunk_blocks(static_cast<std::size_t>(chunk_xz) * chunk_y * chunk_xz);
  std::vector<GbrMesh> meshes(64);
  std::vector<std::uint64_t> ids(64);

  // Submit every chunk, draining finished meshes as they arrive; the drain is
  // what keeps the bounded pipeline from backing up, which is the render-thread
  // behaviour in miniature.
  std::uint64_t next_id = 0;
  for (std::uint32_t cy = 0; cy < chunks_y; ++cy) {
    for (std::uint32_t cz = 0; cz < chunks_x; ++cz) {
      for (std::uint32_t cx = 0; cx < chunks_x; ++cx) {
        const std::uint32_t ox = cx * chunk_xz;
        const std::uint32_t oy = cy * chunk_y;
        const std::uint32_t oz = cz * chunk_xz;
        const std::uint32_t cw = (ox + chunk_xz <= world) ? chunk_xz : (world - ox);
        const std::uint32_t ch = (oy + chunk_y <= height) ? chunk_y : (height - oy);
        const std::uint32_t cd = (oz + chunk_xz <= world) ? chunk_xz : (world - oz);

        extract_chunk(blocks, world, height, ox, oy, oz, cw, ch, cd, chunk_blocks);

        // Retry on back-pressure: the drain below frees room.
        while (gbr_pipeline_submit(g_pipeline, next_id, chunk_blocks.data(), cw, ch, cd) == 1) {
          const int n = gbr_pipeline_drain(g_pipeline, meshes.data(), ids.data(), 64);
          for (int i = 0; i < n; ++i) {
            quads += meshes[i].vertex_count / 4;
            gbr_mesh_free(meshes[i]);
            ++drained;
          }
          if (n == 0) {
            break; // nothing ready yet; the next submit attempt will retry
          }
        }
        ++next_id;
        ++submitted;
      }
    }
  }

  // Drain the rest.
  for (;;) {
    const int n = gbr_pipeline_drain(g_pipeline, meshes.data(), ids.data(), 64);
    if (n <= 0) {
      break;
    }
    for (int i = 0; i < n; ++i) {
      quads += meshes[i].vertex_count / 4;
      gbr_mesh_free(meshes[i]);
      ++drained;
    }
  }

  const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now() - started);

  GbrPipelineStats stats{};
  gbr_pipeline_stats(g_pipeline, &stats);

  log.info("workload: {} chunks of {}x{}x{} meshed in {} us -> {} quads", submitted, chunk_xz,
           chunk_y, chunk_xz, elapsed.count(), quads);
  log.info("pipeline stats: submitted={} meshed={} drained={} dropped={} rejected={}",
           stats.submitted, stats.meshed, stats.drained, stats.dropped, stats.rejected);

  const bool complete = (drained == submitted) && (stats.dropped == 0);
  if (!complete) {
    log.error("workload lost work: submitted {} but drained {} (dropped {})", submitted, drained,
              stats.dropped);
  }
  return complete;
}

} // namespace

namespace pl::config {

template <> struct Schema<GlowberryRustConfig> {
  static constexpr std::string_view title = "Glowberry Rust Core";
  static constexpr std::string_view description =
      "Native acceleration core: SIMD math, greedy meshing, lock-free ring.";

  static constexpr FieldSchema field(std::string_view name) {
    if (name == "version") {
      return {"Version", "Config schema version.", std::nullopt, std::nullopt, true};
    }
    if (name == "enabled") {
      return {"Enabled", "Whether the Rust core is active.", std::nullopt, std::nullopt, false};
    }
    if (name == "pipeline_workers") {
      return {"Pipeline workers", "Worker threads that mesh chunks off the render thread.",
              std::nullopt, std::nullopt, false};
    }
    if (name == "chunk_xz") {
      return {"Chunk footprint", "Chunk size along x and z, in blocks.", std::nullopt,
              std::nullopt, false};
    }
    if (name == "chunk_y") {
      return {"Chunk height", "Chunk size along y, in blocks.", std::nullopt, std::nullopt, false};
    }
    if (name == "world_size") {
      return {"World size", "World footprint along x and z for the startup workload.",
              std::nullopt, std::nullopt, false};
    }
    if (name == "run_workload") {
      return {"Run workload", "Mesh a world through the pipeline at startup.", std::nullopt,
              std::nullopt, false};
    }
    return {"", "", std::nullopt, std::nullopt, false};
  }
};

} // namespace pl::config

/// The mod lifecycle object. The preloader materialises it once and calls the
/// four hooks.
class GlowberryRustMod {
public:
  static GlowberryRustMod &instance() {
    static GlowberryRustMod mod;
    return mod;
  }

  bool load(pl::mod::ModContext &) {
    mConfig.emplace();
    if (!mConfig->load()) {
      log().warn("could not load config.json; using defaults for this session");
      mConfig.reset();
    }

    const auto &cfg = config();
    log().info("glowberry-rust core v{:#010x}", gbr_version());

    const int workers = cfg.pipeline_workers > 0 ? cfg.pipeline_workers : 1;
    g_pipeline = gbr_pipeline_new(static_cast<std::uint32_t>(workers), 256, 256);
    if (g_pipeline == nullptr) {
      log().error("could not start the meshing pipeline (workers {})", workers);
      return false;
    }
    log().info("meshing pipeline ready ({} workers)", workers);

    if (cfg.run_workload && !run_workload(cfg, log())) {
      log().error("startup workload failed; disabling this session");
      return false;
    }
    return true;
  }

  bool enable(pl::mod::ModContext &) {
    if (!config().enabled) {
      log().info("disabled by config; nothing to do");
      return true;
    }
    log().info("enabled");
    return true;
  }

  bool disable(pl::mod::ModContext &) {
    log().info("disabled");
    return true;
  }

  bool unload(pl::mod::ModContext &) {
    if (g_pipeline != nullptr) {
      gbr_pipeline_free(g_pipeline);
      g_pipeline = nullptr;
    }
    log().info("unloaded");
    return true;
  }

private:
  pl::log::Logger &log() { return pl::mod::NativeMod::current()->getLogger(); }

  const GlowberryRustConfig &config() const {
    static const GlowberryRustConfig defaults{};
    return mConfig.has_value() ? mConfig->value() : defaults;
  }

  std::optional<pl::config::ConfigFile<GlowberryRustConfig>> mConfig;
};

// The one export the preloader looks for: the macro emits `PLGetModRegistration`.
PL_REGISTER_MOD(GlowberryRustMod, GlowberryRustMod::instance())
