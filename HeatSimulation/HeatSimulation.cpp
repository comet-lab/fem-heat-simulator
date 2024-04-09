// HeatSimulation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FEM_Simulator.h"

int main()
{
    std::vector<std::vector<std::vector<float>>> Temp = { { {50,50,50}, {50,50,50}, {50,50,50} },
                                                                   { {50,50,50}, {50,0,50}, {50,50,50} },
                                                                   { {50,50,50}, {50,50,50}, {50,50,50} } };
    float tissueSize[3] = { 1.0f,1.0f,1.0f };
    FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 1, 1, 1, 1);
    simulator->deltaT = 0.01f;
    simulator->tFinal = 0.2f;
    int BC[6] = { 0,0,0,0,0,0 };
    simulator->setBoundaryConditions(BC);
    simulator->Jn = 0;
    simulator->setAmbientTemp(0);
    std::vector<std::vector<std::vector<float>>> NFR = { { {0,0,0}, {0,0,0}, {0,0,0} },
                                                                   { {0,0,0}, {0,0,0}, {0,0,0} },
                                                                   { {0,0,0}, {0,0,0}, {0,0,0} } };

    simulator->solveFEA(NFR);
    
    for (int k = 0; k < 3; k++) {
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < 3; i++) {
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
