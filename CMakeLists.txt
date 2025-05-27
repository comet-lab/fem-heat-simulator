# TODO: Review Cal's Changes
cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 23)
project(HeatSimulation)

set(EIGEN_DIR "C:/Users/nepacheco/OneDrive - Worcester Polytechnic Institute (wpi.edu)/PhD/Repositories/CppLibraries/eigen-3.4.0/")  # CHANGE 

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


add_library(HeatSimulation
        src/FEM_Simulator.cpp
        src/FEM_Simulator.h)

# Include in larger project (AEnKF_OOP)
target_include_directories(HeatSimulation PUBLIC include)

# Include local Eigen/Dense header by specifying the full path here
target_include_directories(HeatSimulation PRIVATE ${EIGEN_DIR})

# Add Executable 
add_executable(main 
            src/FEM_Simulator.cpp
            src/FEM_Simulator.h
            "src/main.cpp")

       # Include in larger project (AEnKF_OOP)
target_include_directories(main PUBLIC include)

# Include local Eigen/Dense header by specifying the full path here
target_include_directories(main PRIVATE ${EIGEN_DIR})

enable_testing()

add_executable(FEM_Unit_Tests
  test/FEM_Unit_Tests.cpp
  src/FEM_Simulator.cpp
  src/FEM_Simulator.h
 "test/BaseSim.cpp")
target_include_directories(FEM_Unit_Tests PRIVATE "C:/Users/nepacheco/OneDrive - Worcester Polytechnic Institute (wpi.edu)/PhD/Repositories/CppLibraries/eigen-3.4.0/")
target_link_libraries(FEM_Unit_Tests
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(FEM_Unit_Tests)


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    find_package(Matlab REQUIRED)

    matlab_add_mex(
        NAME MEX_Heat_Simulation
        SRC 
            src/MEX_Heat_Simulation.cpp
            src/FEM_Simulator.cpp
            src/FEM_Simulator.h # <-- Add this
    )
    target_include_directories(MEX_Heat_Simulation PRIVATE
        ${EIGEN_DIR}
    )
    target_compile_options(MEX_Heat_Simulation PRIVATE
        "$<$<CXX_COMPILER_ID:MSVC>:/openmp>"
        "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-fopenmp>"
    )

    target_link_options(MEX_Heat_Simulation PRIVATE
        "$<$<CXX_COMPILER_ID:MSVC>:/openmp>"
        "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-fopenmp>"
    )

    set_target_properties(MEX_Heat_Simulation PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif()