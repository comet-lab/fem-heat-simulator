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
    int nodesPerAxis[3] = { 101,101,100 };

    std::vector<std::vector<std::vector<float>>> Temp(nodesPerAxis[0], std::vector<std::vector<float>>(nodesPerAxis[1], std::vector<float>(nodesPerAxis[2])));
    std::vector<std::vector<std::vector<float>>> FluenceRate(nodesPerAxis[0], std::vector<std::vector<float>>(nodesPerAxis[1], std::vector<float>(nodesPerAxis[2])));
    
    srand(1);

    for (int i = 0; i < nodesPerAxis[0]; i++) {
        for (int j = 0; j < nodesPerAxis[1]; j++) {
            for (int k = 0; k < nodesPerAxis[2]; k++) {
                Temp[i][j][k] = 20;
            }
        }
    }
        
    int Nn1d = 2;
    float tissueSize[3] = { 2.0f,2.0f,1.0f };

    FEM_Simulator simulator(Temp, tissueSize, 0.0062, 4.3, 200, 0.05, Nn1d);

    simulator.alpha = 0.5f;
    simulator.setLayer(0.05f, 30);
    std::cout << "Number of nodes: " << simulator.nodesPerAxis[0] * simulator.nodesPerAxis[1] * simulator.nodesPerAxis[2] << std::endl;
    std::cout << "Number of elems: " << simulator.elementsPerAxis[0] * simulator.elementsPerAxis[1] * simulator.elementsPerAxis[2] << std::endl;
    float laserPose[6] = { 0.0f,0,-35,0,0,0 };
    simulator.setFluenceRate(laserPose, 1, 0.0168);

    std::cout << "Object Created " << std::endl;

    simulator.deltaT = 0.05f;
    int BC[6] = { 2,0,0,0,0,0 };
    simulator.setBoundaryConditions(BC);
    simulator.setFlux(0.0f);
    simulator.setAmbientTemp(24);
    simulator.silentMode = false;
    float totalTime = 1.0f;

    std::vector<std::array<float, 3>> tempSensorLocations = { {0, 0, 0.0}, {0,0,0.95f}, {1,0,0}, {0,1,0},{0,0,1} };
    simulator.setSensorLocations(tempSensorLocations);
    // FEM_Simulator simCopy = FEM_Simulator(simulator);
    std::cout << "Running FEA" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    simulator.setFluenceRate(laserPose, 1, 0.0168);
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
    start = std::chrono::high_resolution_clock::now();
    std::cout << "Initialization Duration: " << duration.count()/1000000.0 << std::endl;
    for (int i = 1; i <= round(totalTime/simulator.deltaT); i++) {
        simulator.setFluenceRate(laserPose, 0.5 + i/10.0, 0.0168);
        simulator.setMUA(i*5 + 20);
        // simulator.parameterUpdate = true;
        simulator.singleStep();
    }
    duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start);
    std::cout << "Time-Stepping Duration: " << duration.count()/1000000.0 << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Top Face Temp: " <<   simulator.Temp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) / 2 * nodesPerAxis[0]) << std::endl;
    std::cout << "Bottom Face Temp: " << simulator.Temp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) / 2 * nodesPerAxis[0] + nodesPerAxis[0]*nodesPerAxis[1]*(nodesPerAxis[2]-1)) << std::endl;
    std::cout << "Front Face Temp: " << simulator.Temp(nodesPerAxis[0] + nodesPerAxis[1] / 2 * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Right Face Temp: " << simulator.Temp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Back Face Temp: " << simulator.Temp((nodesPerAxis[1]-1) / 2 * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Left Face Temp: " <<  simulator.Temp((nodesPerAxis[0]-1) / 2 + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;


#ifdef USE_CUDA
    // GPU Usage
    simulator.setTemp(Temp);// reset temperature
    simulator.setFluenceRate(laserPose, 0, 0.0168); // set fluence rate
    simulator.buildMatrices(); // set fluence rate
    GPUTimeIntegrator gpuHandle(simulator.alpha, simulator.deltaT);
    std::cout << "GPU Handle created " << std::endl;
    gpuHandle.setModel(&simulator);
    std::cout << "Model Assigned" << std::endl;
    gpuHandle.initializeWithModel();
    std::cout << "Model initialized" << std::endl;
    for (int i = 1; i <= round(totalTime/simulator.deltaT); i++) {
        simulator.setFluenceRate(laserPose, 0.5 + i/10.0, 0.0168);
        simulator.setMUA(i*5 + 20);
        gpuHandle.singleStepWithUpdate();
    }
    std::cout << "Top Face Temp: " <<   simulator.Temp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) / 2 * nodesPerAxis[0]) << std::endl;
    std::cout << "Bottom Face Temp: " << simulator.Temp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) / 2 * nodesPerAxis[0] + nodesPerAxis[0]*nodesPerAxis[1]*(nodesPerAxis[2]-1)) << std::endl;
    std::cout << "Front Face Temp: " << simulator.Temp(nodesPerAxis[0] + nodesPerAxis[1] / 2 * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Right Face Temp: " << simulator.Temp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Back Face Temp: " << simulator.Temp((nodesPerAxis[1]-1) / 2 * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    std::cout << "Left Face Temp: " <<  simulator.Temp((nodesPerAxis[0]-1) / 2 + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
#endif
    
    /* Printing Results*/
    if (nodesPerAxis[0] <= 5) {
        for (int k = 0; k < nodesPerAxis[2]; k++) {
            for (int j = 0; j < nodesPerAxis[1]; j++) {
                for (int i = 0; i < nodesPerAxis[0]; i++) {
                    std::cout << simulator.Temp(i + j*nodesPerAxis[0] + k*nodesPerAxis[0]*nodesPerAxis[1]) << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}
