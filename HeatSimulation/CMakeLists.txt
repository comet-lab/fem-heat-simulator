# TODO: Review Cal's Changes
cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)

add_library(HeatSimulation
        FEM_Simulator.cpp
        FEM_Simulator.h
        HeatSimulation.cpp)

# Include in larger project (AEnKF_OOP)
target_include_directories(HeatSimulation PUBLIC include)

# Include local Eigen/Dense header by specifying the full path here
target_include_directories(HeatSimulation PRIVATE /usr/local/Cellar/eigen/3.4.0_1/include/eigen3)