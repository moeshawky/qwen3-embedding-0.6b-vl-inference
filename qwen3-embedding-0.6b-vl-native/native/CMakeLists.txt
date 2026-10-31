cmake_minimum_required(VERSION 3.16)
project(qwen3vl_embed CXX)

# Standalone native (GGML) inference for moeshawky/qwen3-embedding-0.6b-vl.
#
# The GGML backend and the llama.cpp text runtime are NOT vendored into this
# repository: they are consumed from a llama.cpp checkout (pinned upstream
# commit 0a1ef309cdbb9f2024f58a38fb4c42352c5db569). Point LLAMA_CPP_DIR at
# that checkout; the build fails loudly if the expected sources are missing.
set(LLAMA_CPP_DIR "/tmp/opencode/llamacpp" CACHE PATH "llama.cpp source tree providing ggml/ and src/")

if (NOT EXISTS "${LLAMA_CPP_DIR}/ggml/src/ggml.c" OR NOT EXISTS "${LLAMA_CPP_DIR}/src/llama.cpp")
    message(FATAL_ERROR
        "LLAMA_CPP_DIR='${LLAMA_CPP_DIR}' does not look like a llama.cpp checkout.\n"
        "Point it at one, e.g.\n"
        "  cmake -B build -DLLAMA_CPP_DIR=/path/to/llama.cpp")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Only the libraries are needed: no CLI tools, no server, no tests.
set(LLAMA_BUILD_COMMON    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL            OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE           ON  CACHE BOOL "" FORCE)

# The llama.cpp root pulls in ggml and libllama.
add_subdirectory("${LLAMA_CPP_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama")

add_executable(qwen3vl-embed
    src/main.cpp
    src/sidecar.cpp
    src/preprocess.cpp
    src/vision_graph.cpp
    src/text_embed.cpp
    src/npy.cpp
)

target_include_directories(qwen3vl-embed PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor"
)

target_link_libraries(qwen3vl-embed PRIVATE llama)

target_compile_options(qwen3vl-embed PRIVATE
    -Wall -Wextra -Wno-unused-parameter
)

# The vendored stb_image.h is third-party (public domain / Unlicense).
target_compile_definitions(qwen3vl-embed PRIVATE
    STB_IMAGE_IMPLEMENTATION
    STBI_ONLY_PNG
    STBI_ONLY_JPEG
)
