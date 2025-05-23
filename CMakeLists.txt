# TODO: Review Cal's Changes
cmake_minimum_required(VERSION 3.29)
set(CMAKE_CXX_STANDARD 23)
project(HeatSimulation)

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
target_include_directories(HeatSimulation PRIVATE "C:/Users/nepacheco/OneDrive - Worcester Polytechnic Institute (wpi.edu)/PhD/Repositories/CppLibraries/eigen-3.4.0/")

# Add Executable 
add_executable(main 
            src/FEM_Simulator.cpp
            src/FEM_Simulator.h
            "src/main.cpp")

       # Include in larger project (AEnKF_OOP)
target_include_directories(main PUBLIC include)

# Include local Eigen/Dense header by specifying the full path here
target_include_directories(main PRIVATE "C:/Users/nepacheco/OneDrive - Worcester Polytechnic Institute (wpi.edu)/PhD/Repositories/CppLibraries/eigen-3.4.0/")

enable_testing()

add_executable(
  FEM_Unit_Tests
  test/FEM_Unit_Tests.cpp
  src/FEM_Simulator.cpp
  src/FEM_Simulator.h
)
target_include_directories(FEM_Unit_Tests PRIVATE "C:/Users/nepacheco/OneDrive - Worcester Polytechnic Institute (wpi.edu)/PhD/Repositories/CppLibraries/eigen-3.4.0/")
target_link_libraries(
  FEM_Unit_Tests
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(FEM_Unit_Tests)