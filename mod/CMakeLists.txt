cmake_minimum_required(VERSION 3.22)

project(glowberry_rust_mod LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# This mod is built for the same ABI the client ships and the preloader supports.
if(ANDROID AND NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR
            "glowberry_rust_mod supports arm64-v8a only; got ${CMAKE_ANDROID_ARCH_ABI}")
endif()

get_filename_component(GLOWBERRY_RUST_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# A preloader-android checkout supplies the public `pl/` headers and the
# `pl::config` / `pl::log` implementations the mod uses. Point this at one.
set(PRELOADER_ANDROID_ROOT
    "${GLOWBERRY_RUST_ROOT}/third_party/preloader-android"
    CACHE PATH "Path to a preloader-android checkout")

if(NOT EXISTS "${PRELOADER_ANDROID_ROOT}/include/pl/Mod.hpp")
    message(FATAL_ERROR
            "PRELOADER_ANDROID_ROOT must point to a preloader-android checkout. "
            "Missing: ${PRELOADER_ANDROID_ROOT}/include/pl/Mod.hpp")
endif()

# The Rust core, built by `cargo build --release --target <abi>`. Defaults to the
# standard cargo output path for the target ABI.
set(GLOWBERRY_RUST_TARGET "aarch64-linux-android")
set(GLOWBERRY_RUST_LIB
    "${GLOWBERRY_RUST_ROOT}/target/${GLOWBERRY_RUST_TARGET}/release/libglowberry_rust.a"
    CACHE FILEPATH "Path to the Rust static library (libglowberry_rust.a)")

if(NOT EXISTS "${GLOWBERRY_RUST_LIB}")
    message(FATAL_ERROR
            "Rust core not found at ${GLOWBERRY_RUST_LIB}. Build it first: "
            "cargo build --release --target ${GLOWBERRY_RUST_TARGET}")
endif()

add_library(glowberry_rust_mod SHARED GlowberryRustMod.cpp)

target_include_directories(glowberry_rust_mod PRIVATE
    "${PRELOADER_ANDROID_ROOT}/include"
    "${GLOWBERRY_RUST_ROOT}/cpp")

target_compile_options(glowberry_rust_mod PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-Wall -Wextra -Wpedantic>)

set_target_properties(glowberry_rust_mod PROPERTIES
    OUTPUT_NAME "glowberry_rust"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# The mod resolves the preloader's exports from the host process at load time, so
# it links with unresolved symbols allowed rather than against a preloader .so
# (which is not shipped as a linkable artifact). This mirrors the SDK's own
# minimal-mod example.
if(ANDROID)
    target_link_options(glowberry_rust_mod PRIVATE -Wl,--unresolved-symbols=ignore-all)
endif()

# Only `PLGetModRegistration` and the reusable `gbr_*` core stay visible; fmt and
# the C++ glue are local. The map file is the explicit form of the
# `--exclude-libs,ALL` intent above.
target_link_options(glowberry_rust_mod PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-Wl,--gc-sections>
    $<$<PLATFORM_ID:Android>:-Wl,-z,max-page-size=16384>)

if(ANDROID)
    target_link_options(glowberry_rust_mod PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/glowberry_rust.map")
endif()

# The Rust core has no libc++ dependency, but the C++ shim and fmt do.
target_link_libraries(glowberry_rust_mod PRIVATE "${GLOWBERRY_RUST_LIB}")

find_library(GLOWBERRY_LOG_LIB log)
if(GLOWBERRY_LOG_LIB)
    target_link_libraries(glowberry_rust_mod PRIVATE "${GLOWBERRY_LOG_LIB}")
endif()

# The `pl::config` and `pl::log` headers the mod includes pull in these four
# headers. They are fetched at the same pinned commits the preloader uses, so the
# mod compiles against exactly the headers the preloader ships.
include(FetchContent)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03)
FetchContent_Declare(boost_pfr
    GIT_REPOSITORY https://github.com/boostorg/pfr.git
    GIT_TAG 3fe5ce61eee743c6da097c28bc0b84bdf29f6cc4)
FetchContent_Declare(magic_enum
    GIT_REPOSITORY https://github.com/Neargye/magic_enum.git
    GIT_TAG e046b69a3736d314fad813e159b1c192eaef92cd)
FetchContent_Declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 40626af88bd7df9a5fb80be7b25ac85b122d6c21)

set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlohmann_json boost_pfr magic_enum fmt)

target_link_libraries(glowberry_rust_mod PRIVATE
    fmt::fmt
    nlohmann_json::nlohmann_json
    Boost::pfr
    magic_enum::magic_enum)

install(TARGETS glowberry_rust_mod LIBRARY DESTINATION lib)
