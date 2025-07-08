// HeatSimulation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FEM_Simulator.h"

int main()
{
    std::cout << "Starting Program" << std::endl;
    int nodesPerAxis[3] = { 41,41,50 };

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
    simulator.alpha = 1;
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

    std::vector<std::array<float, 3>> tempSensorLocations = { {0, 0, 0.0}, {0,0,0.95f}, {1,0,0}, {0,1,0},{0,0,1} };
    simulator.setSensorLocations(tempSensorLocations);
    FEM_Simulator simCopy = FEM_Simulator(simulator);
    std::cout << "Running FEA" << std::endl;

#ifdef _OPENMP
    std::cout << "OpenMP enabled" << std::endl;
    Eigen::setNbThreads(omp_get_num_procs()/2);
#else
    std::cout << "OpenMP disabled" << std::endl;
    Eigen::setNbThreads(1);
#endif
    std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    float totalTime = 0.05f;
    simulator.silentMode = false;
    simulator.initializeModel();
    for (int i = 0; i < 1; i++) {
        simulator.setFluenceRate(laserPose, 1, 0.0168);
        simulator.multiStep(totalTime);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "FEA Duration: " << duration.count()/1000000.0 << std::endl;
    
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
    else {
        std::cout << "Top Face Temp: " <<   simulator.Temp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) / 2 * nodesPerAxis[0]) << std::endl;
        std::cout << "Bottom Face Temp: " << simulator.Temp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) / 2 * nodesPerAxis[0] + nodesPerAxis[0]*nodesPerAxis[1]*(nodesPerAxis[2]-1)) << std::endl;
        std::cout << "Front Face Temp: " << simulator.Temp(nodesPerAxis[0] + nodesPerAxis[1] / 2 * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
        std::cout << "Right Face Temp: " << simulator.Temp((nodesPerAxis[0]-1) / 2 + (nodesPerAxis[1]-1) * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
        std::cout << "Back Face Temp: " << simulator.Temp((nodesPerAxis[1]-1) / 2 * nodesPerAxis[0] + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
        std::cout << "Left Face Temp: " <<  simulator.Temp((nodesPerAxis[0]-1) / 2 + nodesPerAxis[0] * nodesPerAxis[1] * (nodesPerAxis[2]-1) / 2) << std::endl;
    }

    std::cout << "Sensor Temp: " << simulator.sensorTemps[0][static_cast<int>(totalTime/simulator.deltaT)] << std::endl;
    
}