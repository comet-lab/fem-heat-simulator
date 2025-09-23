cmake_minimum_required(VERSION 3.29)
project(HeatSimulation VERSION 4.0 LANGUAGES CXX)

# ──────────────────────────────────────────────────────────────
# Options & global settings
# ──────────────────────────────────────────────────────────────
option(BUILD_TESTING  "Build unit‑test runner"            ON)
option(BUILD_FEM_MEX  "Build MATLAB MEX wrapper (Release)" ON)
option(USE_CUDA       "Enable GPU solver using AmgX (requires CUDA)" OFF)
set(AMGX_ROOT "/usr/local/amgx" CACHE STRING "Location where AMGX library files are")
# Local install folder
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/HeatSimulationInstall" CACHE PATH "Local install prefix" FORCE)

message(STATUS "AMGX_ROOT: ${AMGX_ROOT}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(CMAKE_CXX_STANDARD 17)

# ──────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────
find_package(Eigen3 3.4 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# Helper to add OpenMP to a target
function(enable_openmp tgt)
    target_link_libraries(${tgt} PUBLIC OpenMP::OpenMP_CXX)
endfunction()

# ──────────────────────────────────────────────────────────────
# CUDA / AmgX support
# ──────────────────────────────────────────────────────────────
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found, enabling GPU and AmgX support")

        find_path(AMGX_INCLUDE_DIR
            NAMES amgx_c.h
            PATHS ${AMGX_ROOT}/include
        )
        find_library(AMGX_LIBRARY
            NAMES amgxsh amgx  # on my install, libamgxsh.so is dynamic and libamgx.a is static
            PATHS ${AMGX_ROOT}/lib
        )
    else()
        set(USE_CUDA OFF)
        message(STATUS "CUDA not found, disabling GPU support")
    endif()
else()
    message(STATUS "USE_CUDA=OFF -- CUDA won't be compiled")
endif()

# ──────────────────────────────────────────────────────────────
# Library: HeatSimulation
# ──────────────────────────────────────────────────────────────

set(HEATSIM_SRC
    src/FEM_Simulator.cpp
)

set(HEATSIM_HEADERS
    include/FEM_Simulator.h
)

# GPU sources (conditionally)
if(USE_CUDA)
    list(APPEND HEATSIM_SRC
        src/GPUTimeIntegrator.cu
        src/AmgXSolver.cpp
    )
    list(APPEND HEATSIM_HEADERS
        include/GPUTimeIntegrator.cuh
        include/AmgXSolver.hpp
    )
endif()


# ──────────────────────────────
# STATIC LIBRARY
# ──────────────────────────────
add_library(HeatSimulationStatic STATIC ${HEATSIM_SRC} ${HEATSIM_HEADERS})
add_library(HeatSimulation::Static ALIAS HeatSimulationStatic)
set_target_properties(HeatSimulationStatic PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ──────────────────────────────
# SHARED LIBRARY
# ──────────────────────────────
add_library(HeatSimulationShared SHARED ${HEATSIM_SRC} ${HEATSIM_HEADERS})
add_library(HeatSimulation::Shared ALIAS HeatSimulationShared)
set_target_properties(HeatSimulationShared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Common include dirs
target_include_directories(HeatSimulationStatic PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<$<BOOL:${USE_CUDA}>:${AMGX_INCLUDE_DIR}>
)
target_include_directories(HeatSimulationShared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    $<$<BOOL:${USE_CUDA}>:${AMGX_INCLUDE_DIR}>
)

# Common compile definitions
if(USE_CUDA)
    foreach(lib HeatSimulationStatic HeatSimulationShared)
        target_compile_definitions(${lib} PUBLIC 
            USE_CUDA
            EIGEN_NO_CUDA
            AMGX_CONFIG_DIR="${CMAKE_SOURCE_DIR}")
    endforeach()
endif()

# Common libraries
set(HEATSIM_LIBS
    Eigen3::Eigen
)
if(USE_CUDA)
    list(APPEND HEATSIM_LIBS
        CUDA::cudart
        ${AMGX_LIBRARY}
        CUDA::cusparse
        CUDA::cusolver
        CUDA::cublas
        dl
    )
endif()

target_link_libraries(HeatSimulationStatic PUBLIC ${HEATSIM_LIBS})
target_link_libraries(HeatSimulationShared PUBLIC ${HEATSIM_LIBS})

# Enable OpenMP for both
enable_openmp(HeatSimulationStatic)
enable_openmp(HeatSimulationShared)

# CUDA standard
set_target_properties(HeatSimulationStatic PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
set_target_properties(HeatSimulationShared PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)


# ──────────────────────────────────────────────────────────────
# Executable: main
# ──────────────────────────────────────────────────────────────
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE HeatSimulation::Shared)

# ──────────────────────────────────────────────────────────────
# Unit tests (GoogleTest via FetchContent)
# ──────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)  # MSVC friendliness
    # We don't need the GTests installing with the library. 
    set(INSTALL_GTEST OFF)
    set(INSTALL_GMOCK OFF)
    FetchContent_MakeAvailable(googletest)

    add_executable(FEM_Unit_Tests
        test/FEM_Unit_Tests.cpp
        test/BaseSim.cpp
    )
    target_link_libraries(FEM_Unit_Tests
        PRIVATE
            HeatSimulation::Shared
            GTest::gtest_main
    )

    if (USE_CUDA)
        add_executable(GPUSolver_Unit_Tests
            test/GPUSolver_Unit_Tests.cpp
            test/BaseGPU.cpp
            test/TestHelpers.hpp
        )
        target_link_libraries(GPUSolver_Unit_Tests
            PRIVATE
                HeatSimulation::Shared
                GTest::gtest_main
        )
    endif()

    include(GoogleTest)
    gtest_discover_tests(FEM_Unit_Tests)
    if (USE_CUDA)
        gtest_discover_tests(GPUSolver_Unit_Tests)
    endif()
endif()

# ──────────────────────────────────────────────────────────────
# MATLAB MEX (only built in Release *and* if MATLAB is found)
# ──────────────────────────────────────────────────────────────
if(BUILD_FEM_MEX AND CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Building MATLAB Executables (MEX)")
    find_package(Matlab REQUIRED)
    if(Matlab_FOUND)
        matlab_add_mex(NAME MEX_Heat_Simulation
            SRC src/MEX_Heat_Simulation.cpp
        )
        target_link_libraries(MEX_Heat_Simulation HeatSimulation::Static)

        set_target_properties(MEX_Heat_Simulation PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        )
        # Manually add OpenMP compile flags (don't use helper function) because MEX requires plain linking
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(MEX_Heat_Simulation PRIVATE -fopenmp)
            target_link_libraries(MEX_Heat_Simulation gomp)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(MEX_Heat_Simulation PRIVATE /openmp)
        endif()


        matlab_add_mex(NAME MEX_Heat_Simulation_MultiStep
            SRC src/MEX_Heat_Simulation_MultiStep.cpp
        )
        target_link_libraries(MEX_Heat_Simulation_MultiStep HeatSimulation::Static)

        set_target_properties(MEX_Heat_Simulation_MultiStep PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
        )
        # Manually add OpenMP compile flags (don't use helper function) because MEX requires plain linking
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(MEX_Heat_Simulation_MultiStep PRIVATE -fopenmp)
            target_link_libraries(MEX_Heat_Simulation_MultiStep gomp)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(MEX_Heat_Simulation_MultiStep PRIVATE /openmp)
        endif()
    else()
        set(BUILD_FEM_MEX OFF)
        message(STATUS "MATLAB not found - not building MATLAB Executable (MEX)")
    endif()
endif()

# ──────────────────────────────────────────────────────────────
# Install rules: library + headers (so other projects can `find_package`)
# ──────────────────────────────────────────────────────────────


# Install headers
install(DIRECTORY include/ DESTINATION include)

install(TARGETS HeatSimulationShared HeatSimulationStatic
        EXPORT HeatSimulationTargets     # <-- this line creates the export set
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

# ────────────── Export targets for find_package ──────────────
# This generates the Targets.cmake file
install(EXPORT HeatSimulationTargets
        FILE HeatSimulationTargets.cmake
        NAMESPACE HeatSimulation::
        DESTINATION lib/cmake/HeatSimulation
)

# Generate the config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/HeatSimulationConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/HeatSimulationConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/HeatSimulation" # relative to install prefix
)

# Install the config file
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/HeatSimulationConfig.cmake"
    DESTINATION "lib/cmake/HeatSimulation"
)


