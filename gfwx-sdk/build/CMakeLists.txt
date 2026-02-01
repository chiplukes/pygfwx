cmake_minimum_required(VERSION 3.15)
project(gfwx_wrapper VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build as shared library
add_library(gfwx SHARED
    gfwx_wrapper.cpp
)

# Include current directory for gfwx.h
target_include_directories(gfwx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Export symbols on Windows
if(WIN32)
    target_compile_definitions(gfwx PRIVATE GFWX_EXPORTS)
endif()

# Enable OpenMP if available (optional, for parallel processing)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gfwx PRIVATE OpenMP::OpenMP_CXX)
endif()
