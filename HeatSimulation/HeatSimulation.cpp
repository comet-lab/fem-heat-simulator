// HeatSimulation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FEM_Simulator.h"

int main()
{

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting Program" << std::endl;

    int nodeSize[3] = { 35,35,51 };

    std::vector<std::vector<std::vector<float>>> Temp(nodeSize[0], std::vector<std::vector<float>>(nodeSize[1], std::vector<float>(nodeSize[2])));
    std::vector<std::vector<std::vector<float>>> NFR(nodeSize[0], std::vector<std::vector<float>>(nodeSize[1], std::vector<float>(nodeSize[2])));
    srand(1);
    for (int i = 0; i < nodeSize[0]; i++) {
        for (int j = 0; j < nodeSize[1]; j++) {
            for (int k = 0; k < nodeSize[2]; k++) {
                Temp[i][j][k] = 20;
                NFR[i][j][k] = 1;
            }
        }
    }
    int Nn1d = 2;
    float tissueSize[3] = { 2.0f,2.0f,0.5f };

    FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.0062, 4.3, 40, 0.05, Nn1d);
    simulator->setLayer(0.1f, 20);
    std::cout << "Number of nodes: " << simulator->nodeSize[0] * simulator->nodeSize[1] * simulator->nodeSize[2] << std::endl;
    std::cout << "Number of elems: " << simulator->gridSize[0] * simulator->gridSize[1] * simulator->gridSize[2] << std::endl;

    std::cout << "Object Created " << std::endl;

    simulator->deltaT = 0.05f;
    simulator->tFinal = 15.0f;
    int BC[6] = { 2,2,2,2,2,2 };
    simulator->setBoundaryConditions(BC);
    simulator->setJn(0);
    simulator->setAmbientTemp(24);
    simulator->setNFR(NFR);

    std::vector<std::array<float, 3>> tempSensorLocations = { {0, 0, 0.4} };
    simulator->setSensorLocations(tempSensorLocations);

    std::cout << "Running FEA" << std::endl;

#ifdef _OPENMP
    Eigen::setNbThreads(omp_get_num_procs()/2);
#else
    Eigen::setNbThreads(1);
#endif
    std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;

    float totalTime = 15.0f;
    simulator->createKMFelem();
    for (int i = 0; i < round(totalTime / simulator->tFinal); i++) {
        
        simulator->performTimeStepping();
    }    
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "FEA Duration: " << duration.count()/1000000.0 << std::endl;

    if (nodeSize[0] <= 5) {
        for (int k = 0; k < nodeSize[2]; k++) {
            for (int j = 0; j < nodeSize[1]; j++) {
                for (int i = 0; i < nodeSize[0]; i++) {
                    std::cout << simulator->Temp[i][j][k] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    else {
        std::cout << "Top Face Temp: " <<   simulator->Temp[nodeSize[0]/2] [nodeSize[1]/2] [0] << std::endl;
        std::cout << "Bottom Face Temp: " << simulator->Temp[nodeSize[0] / 2][nodeSize[1] / 2][nodeSize[2] - 1] << std::endl;
        std::cout << "Front Face Temp: " << simulator->Temp[nodeSize[0]-1][nodeSize[1] / 2][nodeSize[2] / 2] << std::endl;
        std::cout << "Right Face Temp: " << simulator->Temp[nodeSize[0] / 2][nodeSize[1] - 1][nodeSize[2] / 2] << std::endl;
        std::cout << "Back Face Temp: " << simulator->Temp[0][nodeSize[1] / 2][nodeSize[2] / 2] << std::endl;
        std::cout << "Left Face Temp: " <<  simulator->Temp[nodeSize[0]/2] [0] [nodeSize[2]/2] << std::endl;
    }

    std::cout << "Sensor Temp: " << simulator->sensorTemps[0][300] << std::endl;
    
    

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
