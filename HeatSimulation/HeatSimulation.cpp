// HeatSimulation.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "FEM_Simulator.h"

int main()
{
    std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
                                                                   { {0,0,0}, {0,0,0}, {0,0,0} },
                                                                   { {0,0,0}, {0,0,0}, {0,0,0} } };
    float tissueSize[3] = { 1,1,1 };
    FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);
    float xi[3] = { -1,-1,-1 };
    float output = simulator->calculateNA(xi, 0);
    std::cout << output << std::endl;

    std::cout << "Hello World!\n";
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
