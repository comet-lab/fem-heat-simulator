// HeatSimulation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FEM_Simulator.h"

#ifdef USE_CUDA
#include "GPUTimeIntegrator.cuh"
#endif

int main()
{
    std::cout << "Starting Program" << std::endl;
    std::array<float,3> tissueSize = { 2.0f,2.0f,1.0f };
    std::array<long, 3> nodesPerAxis = { 81,81,51 };
    long nNodes = nodesPerAxis[0] * nodesPerAxis[1] * nodesPerAxis[2];
    long nElems = (nodesPerAxis[0] - 1) * (nodesPerAxis[1] - 1) * (nodesPerAxis[2] - 1);
    //std::array<BoundaryType,6> BC = { CONVECTION,CONVECTION,CONVECTION ,CONVECTION ,CONVECTION ,CONVECTION };
    std::array<BoundaryType, 6> BC = { FLUX,FLUX,FLUX ,FLUX ,FLUX ,FLUX};

    std::cout << "Building Mesh" << std::endl;
    Mesh mesh = Mesh::buildCubeMesh(tissueSize, nodesPerAxis, BC);

    float initialTemp = 20.0f;
    Eigen::VectorXf Temp = Eigen::VectorXf::Constant(nNodes, initialTemp);
    Eigen::VectorXf FluenceRate = Eigen::VectorXf::Constant(nNodes, 0);
    
    srand(1);

    float mua = 200;
    float tc = 0.0062;
    float vhc = 4.3;
    float htc = 0.01;
    FEM_Simulator simulator(mua, vhc, tc, htc);
    std::cout << "Setting Mesh" << std::endl;
    simulator.setMesh(mesh);
    simulator.setTemp(Temp);
    std::cout << "Number of nodes: " << nNodes << std::endl;
    std::cout << "Number of elems: " << nElems << std::endl;
    std::cout << "Object Created " << std::endl;
    float totalTime = 1.0f;
    simulator.setAlpha(0.5f);
    simulator.setDt(0.05f);
    simulator.setHeatFlux(0.0);
    simulator.setAmbientTemp(24.0f);

    simulator.silentMode = false;

 
    std::vector<std::array<float, 3>> tempSensorLocations = { {0, 0, 0.0}, {0,0,0.05f}, {0,0,0.5f}, {0,0,0.95f},{0,0,0.5} };
    simulator.setSensorLocations(tempSensorLocations);
    // FEM_Simulator simCopy = FEM_Simulator(simulator);
    std::cout << "Running FEA" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::array<float, 6> laserPose = { 0.0f,0,-35,0,0,0 };
    simulator.setFluenceRate(laserPose, 0, 0.0168);
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    start = std::chrono::high_resolution_clock::now();
    std::cout << "Time to calculate Fluence Rate: " << duration.count()/1000000.0 << std::endl;

    // RUN Solver on CPU
#ifdef _OPENMP
    std::cout << "OpenMP enabled" << std::endl;
    Eigen::setNbThreads(omp_get_num_procs()/2);
#else
    std::cout << "OpenMP disabled" << std::endl;
    Eigen::setNbThreads(1);
#endif
    std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;

 
    simulator.initializeModel();   
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "Initialization Duration: " << duration.count()/1000000.0 << std::endl;
    start = std::chrono::high_resolution_clock::now();
    // for (int i = 1; i <= round(totalTime/simulator.deltaT); i++) {
    //     // simulator.setFluenceRate(laserPose, 0.5 + i/10.0, 0.0168);
    //     // simulator.setMUA(i*5 + 20);
    //     // simulator.parameterUpdate = true;
    //     simulator.singleStep();
    // }
    simulator.multiStep(totalTime);
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "Time-Stepping Duration: " << duration.count()/1000000.0 << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    Eigen::VectorXf outputTemp = simulator.Temp();

    std::cout << "Top Face Temp: " << outputTemp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) / 2 * nodesPerAxis[0]) << std::endl;
    std::cout << "Bottom Face Temp: " << outputTemp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) / 2 * nodesPerAxis[0] + nodesPerAxis[0]*nodesPerAxis[1]*(nodesPerAxis[2]-1)) << std::endl;
    std::cout << "Front Face Temp: " << outputTemp(nodesPerAxis[0] + nodesPerAxis[1] / 2 * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Right Face Temp: " << outputTemp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Back Face Temp: " << outputTemp((nodesPerAxis[1]-1) / 2 * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Left Face Temp: " << outputTemp((nodesPerAxis[0]-1) / 2 + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;


#ifdef USE_CUDA
    // GPU Usage
    simulator.setTemp(Temp_);// reset temperature
    simulator.setFluenceRate(laserPose, 0, 0.0168); // set fluence rate
    simulator.buildMatrices(); // set fluence rate
    GPUTimeIntegrator gpuHandle(simulator.alpha_, simulator.dt_);
    std::cout << "\nGPU Handle created " << std::endl;
    gpuHandle.setModel(&simulator);
    std::cout << "Model Assigned" << std::endl;
    gpuHandle.initializeWithModel();
    simulator.initializeSensorTemps(round(totalTime/simulator.dt_));
    simulator.updateTemperatureSensors(0);
    start = std::chrono::high_resolution_clock::now();
    std::cout << "Model initialized" << std::endl;
    for (int i = 1; i <= round(totalTime/simulator.dt_); i++) {
        simulator.setFluenceRate(laserPose, 0.5 + i/10.0, 0.0168);
        simulator.setMUA(i*5 + 20);
        gpuHandle.singleStepWithUpdate();
        simulator.updateTemperatureSensors(i);
    }
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "Time-Stepping Duration GPU: " << duration.count()/1000000.0 << std::endl;
    start = std::chrono::high_resolution_clock::now();

    Eigen::VectorXf outputTemp = simulator.Temp();
    std::cout << "Top Face Temp: " <<  outputTemp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) / 2 * nodesPerAxis[0]) << std::endl;
    std::cout << "Bottom Face Temp: " << outputTemp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) / 2 * nodesPerAxis[0] + nodesPerAxis[0]*nodesPerAxis[1]*(nodesPerAxis[2]-1)) << std::endl;
    std::cout << "Front Face Temp: " << outputTemp(nodesPerAxis[0] + nodesPerAxis[1] / 2 * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Right Face Temp: " << outputTemp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Back Face Temp: " << outputTemp((nodesPerAxis[1]-1) / 2 * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Left Face Temp: " << outputTemp((nodesPerAxis[0]-1) / 2 + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
#endif
    
    /* Printing Results*/
    if (nodesPerAxis[0] <= 5) {
        for (int k = 0; k < nodesPerAxis[2]; k++) {
            for (int j = 0; j < nodesPerAxis[1]; j++) {
                for (int i = 0; i < nodesPerAxis[0]; i++) {
                    std::cout << simulator.Temp()(i + j*nodesPerAxis[0] + k*nodesPerAxis[0]*nodesPerAxis[1]) << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
