cmake_minimum_required(VERSION 3.29)
project(HeatSimulation VERSION 0.1 LANGUAGES CXX)

# ──────────────────────────────────────────────────────────────
# Options & global settings
# ──────────────────────────────────────────────────────────────
option(BUILD_TESTING  "Build unit‑test runner"            ON)
option(BUILD_MEX      "Build MATLAB MEX wrapper (Release)" ON)

include(GNUInstallDirs)                 # Gives ${CMAKE_INSTALL_*} variables
set(CMAKE_CXX_STANDARD 23)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ──────────────────────────────────────────────────────────────
# Dependencies that come from the system or FetchContent
# ──────────────────────────────────────────────────────────────
find_package(Eigen3 3 CONFIG REQUIRED)
find_package(OpenMP       REQUIRED)

# Helper to add OpenMP to any target with one line
function(enable_openmp tgt)
    target_link_libraries(${tgt} PUBLIC OpenMP::OpenMP_CXX)
endfunction()

# ──────────────────────────────────────────────────────────────
# Library: HeatSimulation
# ──────────────────────────────────────────────────────────────
set(HEATSIM_SRC
    src/FEM_Simulator.cpp
)

set(HEATSIM_HEADERS
    include/FEM_Simulator.h
)

add_library(HeatSimulation STATIC ${HEATSIM_SRC} ${HEATSIM_HEADERS})
add_library(HeatSimulation::HeatSimulation ALIAS HeatSimulation)

target_compile_options(HeatSimulation PUBLIC -fPIC) # This is needed because linux treats mex files as libraries
#target_compile_features(HeatSimulation PUBLIC cxx_std_23)
target_include_directories(HeatSimulation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(HeatSimulation PUBLIC Eigen3::Eigen)
enable_openmp(HeatSimulation)

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

    include(GoogleTest)
    gtest_discover_tests(FEM_Unit_Tests)
endif()

# ──────────────────────────────────────────────────────────────
# MATLAB MEX (only built in Release *and* if MATLAB is found)
# ──────────────────────────────────────────────────────────────
if(BUILD_MEX AND CMAKE_BUILD_TYPE STREQUAL "Release")
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
