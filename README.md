# HeatSimulation


This code base models the thermal response of tissue to laser irradiation using the finite element method (FEM).
Our model can handle arbitrary meshes made from either linear hexahedral, linear tetrahedral, or quadratic tetrahedral elements.
Each element face along the mesh boundary can have one of three conditions: a flux boundary, a convection boundary, or a heat sink. 
For modeling details, see the reference below [1]. 

The main FEM modeling is performed in C++ using the Eigen Library. The file 'main.cpp' provides an example of 
how to initialize and run an FEM_Simulation object, which is defined in 'FEM_Simulation.h'. If you would like to run the
simulations on a GPU, simply run the enableGPU() function. Additionally, the simulation can be called and run 
from MATLAB using the MEX file executable that gets generated 

## Dependencies

Eigen3 Version 3.4 must be installed.

### For GPU Acceleration

CudaToolkit Version 12.8 must be installed 

AMGX must be installed. 

## Building the project

To build the project run the following commands:

``mkdir build``\
``cd build``\
``cmake -DCMAKE_BUILD_TYPE=Release ..``\
``cmake --build .``

This copmile a library that can be included in other projects. Additionally, it will compile a file called main.exe which serves as a test script. 

### cmake arguments

- ``CMAKE_BUILD_TYPE``: can be ``Release`` (which is necessary to build MEX files) or ``Debug``. 
- ``BUILD_TESTING``: can be ``ON`` (Default) or ``OFF``. Determines if test executables are created (FEM_Unit_Tests.exe and GPUSolver_Unit_Tests.exe)
- ``BUILD_FEM_MEX``: can be ``ON`` (Default) or ``OFF``. Determines if the MEX files (MEX_Heat_Simulation.mex*64 and MEX_Heat_Simulation_MultiStep.mex*64) are compiled
- ``USE_CUDA``: can be ``ON`` or ``OFF`` (Default). Determines whether to compile the code necessary for GPU acceleration. 
- ``AMGX_ROOT``: The root location of the AMGX library files. Default is /usr/local/amgx. Necessary if ``USE_CUDA`` is ``ON``.

### Library Install
The library name is HeatSimulation. If you would like to install the library so it can be found using
``find-package(HeatSimulation)`` then run the following command:

``cmake --install .``

## Description of the simulation class

The simulation class is declared in FEM_Simulator.h and defined in FEM_Simulator.cpp. 
See 'main.cpp' for an example of how to initialize and run the simulator. The ``FEM_Simulator`` requires a mesh object to enable simulations.
To define a simple cuboid mesh, use the static method part of the ``Mesh`` class with a vector of x, y, and z coordinates
for the nodes. Otherwise, define a mesh with a list of node coordinates, a list of elements, and a list of boundary faces. 

The mesh is allowed to define 3 different boundary types:

- 0: Flux Boundary (for a flux boundary, the ``flux`` attribute needs to be set)
- 1: Convection Boundary (for a Convection boundary, the ``ambientTemp`` attribute needs to be set)
- 2: Heat Sink (the initial temperature set at the boundary will be held constant)

In addition to defining the mesh, the user must set tissue specific parameters, boundary conditions, and time-stepping parameters
listed below.

To construct a basic simulation object, you need the following information

- ``TC``: the thermal conductivity of the tissue [ $\frac{\text{W}}{\text{cm}~^o\text{C}}$ ]
- ``VHC``: the volumetric heat capacity of the tissue [ $\frac{\text{J}}{\text{cm}^3~^o\text{C}}$ ]
- ``MUA``: the absorption coefficient of the laser [ $\text{cm}^{-1}$ ]
- ``HTC``: the heat transfer coefficient [ $\frac{\text{W}}{\text{cm}^2~^o\text{C}}$ ]
- ``Temp``: the initial temperature at every node in the mesh [ $^o\text{C}$ ],
- ``fluenceRate``: the volumetric power density at each node or in each element [ $\text{W}/{\text{cm}^3}$ ]
- ``dt``: the time step duration
- ``alpha``: controls the balance between implicit and explicit integration. 1: Fully Implicit, 0: Fully Explicit. 

If the mesh contains flux or convection boundaries, some additional parameters will need to be set:

- ``flux``: the heat flux entering the element at the boundary
- ``ambientTemp``: the ambient temperature of the fluid surrounding the mesh for convection

Additionally print statements can be controlled with the, ``silentMode`` attribute. If you would like to enable multi-threading,
simply call ``Eigen::SetNbThreads()`` before running the simulator. 

### Running the Simulator on CPU
After all appropriate conditions have been set for the object, we can initialize and run the model. 
To initialize the model, call the function ``FEM_Simulator::initializeModel()``. This will perform the spatial discretization and 
build the global matrices. Additionally, it will initialize the integration scheme. Then call either ``FEM_Simulator::singleStep()``
or ``FEM_Simulator::multiStep()`` to simulate the time stepping. If the mesh geometry or boundary conditions change between step calls,
the system will have to be reinitialized. If you would like to reset the time integration, simply call ``FEM_Simulator::initializeTimeIntegration()``. 

### Using GPU Acceleration
To enable GPU acceleration, simply call ``FEM_Simualtor::enableGPU()`` prior to calling ``FEM_Simulator::initializeModel()``. If you want to 
switch between GPU and CPU, simply call ``FEM_Simulator::initializeTimeIntegration()`` after enabling or disabling the GPU. 

## MATLAB Usage
The C++ code is callable through MATLAB via the MEX file 'MEX_Heat_Simulation.mex*64'. This file is built be default when building the code in 
Release mode. To make interfacing with the MEX function smoother, we have defined a HeatSimulator object in MATLAB which serves as an
in-between. Examples code for using this interface is found in the folder MATLAB/TestCode.


# References
[1] N. E. Pacheco, K. Zhang, A. S. Reyes, C. J. Pacheco, L. Burstein,
and L. Fichera, ''Towards a physics engine to simulate robotic laser
surgery: Finite element modeling of thermal laser-tissue interactions'',
International Symposium on Medical Robotics, 2025, [In Press] Available:
https://arxiv.org/abs/2411.14249.
