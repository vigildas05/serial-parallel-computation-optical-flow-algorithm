cmake_minimum_required(VERSION 3.24)
project(opt-flow-parallel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)

# MSVC's default OpenMP runtime is OpenMP 2.0 (no collapse()); use its LLVM runtime instead.
set(OpenMP_RUNTIME_MSVC "llvm")
find_package(OpenMP REQUIRED COMPONENTS CXX)

# ---------------------------------------------------------------------------
# Optional CUDA support
#   -DOFP_ENABLE_CUDA=OFF            force a CPU-only build
#   -DCMAKE_CUDA_ARCHITECTURES=86    override the target GPU architecture
#                                    (defaults to the GPU(s) on the build machine)
# ---------------------------------------------------------------------------
option(OFP_ENABLE_CUDA "Build the CUDA implementation if a CUDA compiler is available" ON)
set(OFP_HAVE_CUDA OFF)

if(OFP_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        # Must be set before enable_language(CUDA), otherwise CMake picks its own default.
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        set(OFP_HAVE_CUDA ON)
        message(STATUS "CUDA enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
    else()
        message(STATUS "No CUDA compiler found - building CPU-only")
    endif()
else()
    message(STATUS "CUDA disabled by OFP_ENABLE_CUDA=OFF - building CPU-only")
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
file(GLOB CPU_SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_executable(opt-flow-parallel main.cpp ${CPU_SOURCES})
target_include_directories(opt-flow-parallel PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opt-flow-parallel PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

if(MSVC)
    target_compile_options(opt-flow-parallel PRIVATE /W3 /utf-8)
else()
    target_compile_options(opt-flow-parallel PRIVATE -Wall -Wextra)
endif()

if(OFP_HAVE_CUDA)
    file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS src/cuda/*.cu src/cuda/*.cpp)
    if(CUDA_SOURCES)
        add_library(hs_cuda STATIC ${CUDA_SOURCES})
        target_include_directories(hs_cuda PUBLIC include PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(hs_cuda PUBLIC CUDA::cudart ${OpenCV_LIBS})
        target_compile_options(hs_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)   # source-level info for Nsight Compute
        target_compile_definitions(opt-flow-parallel PRIVATE OFP_HAVE_CUDA=1)
        target_link_libraries(opt-flow-parallel PRIVATE hs_cuda)
    endif()
endif()

# On Windows, copy the OpenCV DLLs next to the executable so it runs from the build dir.
if(WIN32)
    add_custom_command(TARGET opt-flow-parallel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:opt-flow-parallel> $<TARGET_FILE_DIR:opt-flow-parallel>
        COMMAND_EXPAND_LISTS)
endif()
