#include <gtest/gtest.h>
#include "ShapeFunctions/HexLinear.hpp"
#include <iostream>
#include <string>


/**
* Tests the 3D tri-linear shape function for hexahedral elements.
* We are enforcing the order here as well that
* A[0] = (0,0,0)
* A[1] = (1,0,0)
* A[2] = (0,1,0)
* A[3] = (1,1,0)
* A[4] = (0,0,1) ... etc.
*/
TEST(HexLinear, testN)
{
	ShapeFunctions::HexLinear testFunctions; 
	int Nne = 8; // 8 nodes in hexahedral elements with linear shape functions
	// xi1 has the coordinates at the nodal locations in the element
	std::array<std::array<float, 3>, 8> xi1 = { { {-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1},
												  {-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1} } };
	// xi2 has the coordinates of a different node
	std::array<std::array<float, 3>, 8> xi2 = { { {1,1,1},{-1,1,1},{1,-1,1},{-1,-1,1},
												  {1,1,-1},{-1,1,-1},{1,-1,-1},{-1,-1,-1} } };
	for (int Ai = 0; Ai < Nne; Ai++) {
		float output1 = testFunctions.N(xi1[Ai], Ai);
		float output2 = testFunctions.N(xi2[Ai], Ai);
		float output3 = testFunctions.N({ 0,0,0 }, Ai);

		EXPECT_FLOAT_EQ(1.0f, output1);
		EXPECT_FLOAT_EQ(0.0f, output2);
		EXPECT_FLOAT_EQ(1 / 8.0f, output3);
	}
}

/*
* Testing the derivative of the 3D shape function for hexahedral elements
* Same A layout as before. We treat it as a binary of from x,y,z
*/
TEST(HexLinear, testdNdxi)
{
	ShapeFunctions::HexLinear testFunctions;
	int Nn1d = 2;
	int Nne = 8;
	std::array<std::array<float, 3>, 8> truthTable = { { {-1 / 2.0f,-1 / 2.0f,-1 / 2.0f}, {1 / 2.0f,-1 / 2.0f,-1 / 2.0f},
														 {-1 / 2.0f,1 / 2.0f,-1 / 2.0f},{1 / 2.0f,1 / 2.0f,-1 / 2.0f},
														 {-1 / 2.0f,-1 / 2.0f,1 / 2.0f},{1 / 2.0f,-1 / 2.0f,1 / 2.0f},
														 {-1 / 2.0f,1 / 2.0f,1 / 2.0f},{1 / 2.0f,1 / 2.0f,1 / 2.0f} } };

	std::array<std::array<float, 3>, 8> xi1 = { { {-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1},
												  {-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1} } };

	for (int Ai = 0; Ai < Nne; Ai++)
	{
		Eigen::Vector3f output1 = testFunctions.dNdxi(xi1[Ai], Ai);
		Eigen::Vector3f output2 = testFunctions.dNdxi({ 0.0f,0.0f,0.0f }, Ai);

		for (int i = 0; i < 3; i++)
		{
			EXPECT_FLOAT_EQ(truthTable[Ai][i], output1(i)) << "Ai: " << Ai << " i: " << i;
			EXPECT_FLOAT_EQ(truthTable[Ai][i] / 4.0f, output2(i)) << "Ai: " << Ai << " i: " << i;
		}
	}
}

TEST(HexLinear, testN_face)
{
	ShapeFunctions::HexLinear testFunctions;


}