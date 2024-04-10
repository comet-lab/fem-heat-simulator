// HeatSimulation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FEM_Simulator.h"

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Starting Program" << std::endl;
    int nodeSize[3] = { 10,10,10};
    std::vector<std::vector<std::vector<float>>> Temp(nodeSize[0], std::vector<std::vector<float>>(nodeSize[1], std::vector<float>(nodeSize[2])));
    std::vector<std::vector<std::vector<float>>> NFR(nodeSize[0], std::vector<std::vector<float>>(nodeSize[1], std::vector<float>(nodeSize[2])));
    for (int i = 0; i < nodeSize[0]; i++) {
        for (int j = 0; j < nodeSize[1]; j++) {
            for (int k = 0; k < nodeSize[2]; k++) {
                Temp[i][j][k] = 0.0f;
                NFR[i][j][k] = 0.0f;
            }
        }
    }

    float tissueSize[3] = { 1.0f,1.0f,1.0f };
    FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.0062, 5.22, 100, 1);

    std::cout << "Object Created " << std::endl;

    simulator->deltaT = 0.01f;
    simulator->tFinal = 0.2f;
    int BC[6] = { 0,0,0,0,0,0 };
    simulator->setBoundaryConditions(BC);
    simulator->setJn(0);
    simulator->setAmbientTemp(0);
    
    std::cout << "Running FEA" << std::endl;
    simulator->solveFEA(NFR);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "FEA Duration: " << duration.count()/1000000.0 << std::endl;

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

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
