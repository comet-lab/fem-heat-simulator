cmake_minimum_required(VERSION 3.29)
project(HeatSimulation VERSION 3.2 LANGUAGES CXX CUDA)

# ──────────────────────────────────────────────────────────────
# Options & global settings
# ──────────────────────────────────────────────────────────────
option(BUILD_TESTING  "Build unit‑test runner"            ON)
option(BUILD_FEM_MEX      "Build MATLAB MEX wrapper (Release)" OFF)
option(USE_AMGX "Enable GPU solver using AmgX (requires CUDA)" ON)


include(GNUInstallDirs)                 # Gives ${CMAKE_INSTALL_*} variables
set(CMAKE_CXX_STANDARD 17)              # CMAKE_CXX_STANDARD is 17 because we need to use GCC 9.4 and CUDA 12.8
find_package(CUDAToolkit)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ──────────────────────────────────────────────────────────────
# Dependencies that come from the system or FetchContent
# ──────────────────────────────────────────────────────────────
find_package(Eigen3 3.4 CONFIG REQUIRED)
find_package(OpenMP       REQUIRED)

# Helper to add OpenMP to any target with one line
function(enable_openmp tgt)
    target_link_libraries(${tgt} PUBLIC OpenMP::OpenMP_CXX)
endfunction()

# ──────────────────────────────────────────────────────────────
# Enabling GPU Support if possible
# ──────────────────────────────────────────────────────────────
if(USE_AMGX)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA found, enabling AmgX support")
        add_definitions(-DUSE_AMGX)

        # On Linux: AmgX ships as amgxsh.so and amgx_c.h
        # On Windows: amgxsh.dll + amgx.lib (import library)
        # Adjust paths if needed, or set AMGX_ROOT manually
        find_path(AMGX_INCLUDE_DIR amgx_c.h
            HINTS ${CUDAToolkit_INCLUDE_DIRS}
        )
        find_library(AMGX_LIBRARY NAMES amgxsh
            HINTS ${CUDAToolkit_LIBRARY_DIR}
        )
        # Path to AmgX installation
        set(AMGX_ROOT "/usr/local/amgx")  # or your CMAKE_INSTALL_PREFIX

        # Find the include directory
        find_path(AMGX_INCLUDE_DIR
            NAMES amgx_c.h
            PATHS ${AMGX_ROOT}/include
        )

        # Find the library
        find_library(AMGX_LIBRARY
            NAMES amgx amgxsh
            PATHS ${AMGX_ROOT}/lib
        )
    else()
        message(STATUS "CUDA not found, disabling AmgX")
        set(USE_AMGX OFF)
    endif()
endif()

# ──────────────────────────────────────────────────────────────
# Library: HeatSimulation
# ──────────────────────────────────────────────────────────────
# Enable CUDA separable compilation (needed if multiple .cu files call each other)
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Library sources
set(HEATSIM_SRC
    src/FEM_Simulator.cpp
    src/AmgXSolver.cpp
    src/GPUSolver.cu   # CUDA source
)

set(HEATSIM_HEADERS
    include/FEM_Simulator.h
    include/AmgXSolver.hpp
    include/GPUSolver.cuh
)

# Create the library with CUDA support
add_library(HeatSimulation STATIC ${HEATSIM_SRC} ${HEATSIM_HEADERS})
add_library(HeatSimulation::HeatSimulation ALIAS HeatSimulation)
set_target_properties(HeatSimulation PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Enable PIC
target_compile_options(HeatSimulation PUBLIC -fPIC)

# Include directories
target_include_directories(HeatSimulation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        ${AMGX_INCLUDE_DIR} 
)

# Compile definitions
if(USE_AMGX)
    target_compile_definitions(HeatSimulation PUBLIC USE_AMGX)
endif()

# Link libraries
target_link_libraries(HeatSimulation
    PUBLIC
        Eigen3::Eigen
        CUDA::cudart
        ${AMGX_LIBRARY}
        CUDA::cusparse
        CUDA::cusolver
        CUDA::cublas
        dl
)

# Enable OpenMP
enable_openmp(HeatSimulation)

# Set CUDA standard explicitly (optional but recommended)
set_target_properties(HeatSimulation PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)


# ──────────────────────────────────────────────────────────────
# Executable: main
# ──────────────────────────────────────────────────────────────
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE HeatSimulation::HeatSimulation)

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
            HeatSimulation::HeatSimulation
            GTest::gtest_main
    )

    add_executable(GPUSolver_Unit_Tests
        test/GPUSolver_Unit_Tests.cpp
        test/BaseGPU.cpp
    )
    target_link_libraries(GPUSolver_Unit_Tests
        PRIVATE
            HeatSimulation::HeatSimulation
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(FEM_Unit_Tests)
    gtest_discover_tests(GPUSolver_Unit_Tests)
endif()

# ──────────────────────────────────────────────────────────────
# MATLAB MEX (only built in Release *and* if MATLAB is found)
# ──────────────────────────────────────────────────────────────
if(BUILD_FEM_MEX AND CMAKE_BUILD_TYPE STREQUAL "Release")
    find_package(Matlab REQUIRED)
    if(Matlab_FOUND)
        matlab_add_mex(NAME MEX_Heat_Simulation
            SRC src/MEX_Heat_Simulation.cpp
        )
        target_link_libraries(MEX_Heat_Simulation HeatSimulation::HeatSimulation)

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
        target_link_libraries(MEX_Heat_Simulation_MultiStep HeatSimulation::HeatSimulation)

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
    endif()
endif()

# ──────────────────────────────────────────────────────────────
# Install rules: library + headers (so other projects can `find_package`)
# ──────────────────────────────────────────────────────────────
install(TARGETS HeatSimulation
        EXPORT HeatSimulationTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT HeatSimulationTargets
        FILE HeatSimulationTargets.cmake
        NAMESPACE HeatSimulation::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HeatSimulation)
