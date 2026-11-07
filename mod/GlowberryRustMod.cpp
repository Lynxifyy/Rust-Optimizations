// GlowberryClient native mod: the Rust acceleration core.
//
// This is a preloader mod, not an APK component. The preloader `dlopen`s the
// built `.so` and calls the lifecycle below; the `.so` exports
// `PLGetModRegistration` (emitted by `PL_REGISTER_MOD`) and, because it links
// the Rust static library, the whole `gbr_*` C ABI as well.
//
// What it does on load:
//   1. Resolves the version of the linked Rust core and logs it, so a session's
//      log line proves which core actually loaded.
//   2. Creates the lock-free ring the render thread will drain.
//   3. Meshes a self-test chunk through the Rust core and checks the result, so
//      a broken core is reported at load instead of silently producing no
//      geometry later.
//
// What it deliberately does NOT do:
//   - It does not install a hook. The game's chunk mesher is not reachable
//     (libminecraftpe.so is stripped; see the client's
//     `docs/agents/native-entity-camera-feed-what-is-and-is-not-reachable.md`),
//     so a hook here would be guessing an address. The core is a library the
//     client's own voxel work calls, and this mod is the proof it loads and runs
//     inside the game process.
//   - It does not touch malloc/free. Allocator interposition was removed from the
//     preloader for concrete reasons (hidden visibility meant the definitions
//     never reached the game's allocations; thread_local recurses under the
//     emulated TLS of minSdk 28; a libc block freed by the interposer faults) and
//     must not be reintroduced. See the preloader's AGENTS.md.
//
// Build: `mod/CMakeLists.txt`, cross-compiled for arm64-v8a by the
// `Build .so Mod` workflow.

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
  /// Ring capacity for queued meshes; rounded up to a power of two by the core.
  int ring_capacity = 256;
  /// Worker threads for a batch mesh; 0 or 1 meshes inline on the caller.
  int mesh_workers = 2;
  /// Run the load-time self-test through the Rust core.
  bool self_test = true;
};

/// Runs the load-time self-test: mesh a 16^3 chunk with one solid block and check
/// the core returns six quads (24 vertices). Returns true when the core is sane.
bool run_self_test(pl::log::Logger &log) {
  constexpr std::uint32_t kSize = 16;
  std::vector<std::uint8_t> blocks(
      static_cast<std::size_t>(kSize) * kSize * kSize, 0);
  // One solid block at (1,1,1), x-fastest layout.
  blocks[1 + kSize * (1 + kSize * 1)] = 1;

  GbrMesh mesh = gbr_mesh_blocks(blocks.data(), kSize, kSize, kSize);
  const bool ok = mesh.vertex_count == 24 && mesh.index_count == 36;
  if (!ok) {
    log.error("self-test failed: expected 24 vertices, got {}",
              mesh.vertex_count);
  }
  gbr_mesh_free(mesh);
  return ok;
}

} // namespace

namespace pl::config {

template <> struct Schema<GlowberryRustConfig> {
  static constexpr std::string_view title = "Glowberry Rust Core";
  static constexpr std::string_view description =
      "Native acceleration core: SIMD math, greedy meshing, lock-free ring.";

  static constexpr FieldSchema field(std::string_view name) {
    if (name == "version") {
      return {"Version", "Config schema version.", std::nullopt, std::nullopt,
              true};
    }
    if (name == "enabled") {
      return {"Enabled", "Whether the Rust core is active.", std::nullopt,
              std::nullopt, false};
    }
    if (name == "ring_capacity") {
      return {"Ring capacity",
              "Max queued meshes before the ring back-pressures.", std::nullopt,
              std::nullopt, false};
    }
    if (name == "mesh_workers") {
      return {"Mesh workers", "Threads for a batch mesh; 0 or 1 runs inline.",
              std::nullopt, std::nullopt, false};
    }
    if (name == "self_test") {
      return {"Self-test", "Verify the core at load time.", std::nullopt,
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

    mRing = gbr_ring_new(static_cast<std::uint32_t>(cfg.ring_capacity));
    if (mRing == nullptr) {
      log().error("could not allocate the mesh ring (capacity {})",
                  cfg.ring_capacity);
      return false;
    }
    log().info("mesh ring ready, capacity {}", cfg.ring_capacity);

    if (cfg.self_test && !run_self_test(log())) {
      log().error("core self-test failed; disabling this session");
      return false;
    }
    return true;
  }

  bool enable(pl::mod::ModContext &) {
    if (!config().enabled) {
      log().info("disabled by config; nothing to do");
      return true;
    }
    log().info("enabled (mesh_workers={})", config().mesh_workers);
    return true;
  }

  bool disable(pl::mod::ModContext &) {
    log().info("disabled");
    return true;
  }

  bool unload(pl::mod::ModContext &) {
    if (mRing != nullptr) {
      gbr_ring_free(mRing);
      mRing = nullptr;
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
  GbrRing *mRing{};
};

// The one export the preloader looks for: the macro emits `PLGetModRegistration`.
PL_REGISTER_MOD(GlowberryRustMod, GlowberryRustMod::instance())
