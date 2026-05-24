cmake_minimum_required(VERSION 3.16)
project(voxtral_rust LANGUAGES C CXX)

find_program(CARGO cargo REQUIRED)

set(RUST_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(RUST_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/rust_target)
set(VOXTRAL_LIB_PATH ${RUST_BUILD_DIR}/release/libvoxtral.so)

add_custom_command(
        OUTPUT ${VOXTRAL_LIB_PATH}
        COMMAND ${CMAKE_COMMAND} -E echo "Building voxtral-mini-realtime (cdylib)..."
        COMMAND ${CARGO} build --release
        --target-dir ${RUST_BUILD_DIR}
        --features wgpu
        --package voxtral-mini-realtime
        WORKING_DIRECTORY ${RUST_SOURCE_DIR}
        COMMENT "Building Voxtral Rust cdylib (release)..."
        VERBATIM
        USES_TERMINAL
        DEPENDS ${RUST_SOURCE_DIR}/Cargo.toml
)

add_custom_target(voxtral_rust_build ALL DEPENDS ${VOXTRAL_LIB_PATH})

add_library(voxtral_rust SHARED IMPORTED GLOBAL)
add_dependencies(voxtral_rust voxtral_rust_build)

set_target_properties(voxtral_rust PROPERTIES
        IMPORTED_LOCATION "${VOXTRAL_LIB_PATH}"
)

target_include_directories(voxtral_rust INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

message(STATUS "Voxtral .so path: ${VOXTRAL_LIB_PATH}")