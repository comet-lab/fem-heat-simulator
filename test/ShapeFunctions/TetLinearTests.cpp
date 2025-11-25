#include <gtest/gtest.h>
#include "ShapeFunctions/TetLinear.hpp"
#include <iostream>
#include <string>

TEST(TetLinear, testN)
{
	ShapeFunctions::TetLinear testFunctions;
	int Nne = 4; // 4 nodes in hexahedral elements with linear shape functions
	// xi1 has the coordinates at the nodal locations in the element
	std::array<std::array<float, 3>, 4> xi1 = { { {0,0,0},{1,0,0},{0,1,0},{0,0,1} } };
	// xi2 has the coordinates of a different node
	std::array<std::array<float, 3>, 4> xi2 = { { {1,0,0},{0,1,0},{0,0,1},{0,0,0} } };

	for (int Ai = 0; Ai < Nne; Ai++) {
		float output1 = testFunctions.N(xi1[Ai], Ai);
		float output2 = testFunctions.N(xi2[Ai], Ai);
		float output3 = testFunctions.N({ 1 / 3.0f, 1 / 3.0f, 1 / 3.0f }, Ai);

		EXPECT_FLOAT_EQ(1.0f, output1);
		EXPECT_FLOAT_EQ(0.0f, output2);
		if (Ai == 0)
			EXPECT_NEAR(0.0f, output3, 0.000001);
		else
			EXPECT_FLOAT_EQ(1 / 3.0f, output3);
	}
}

/*
* Testing the derivative of the 3D shape function for tetrahedral elements
* Same A layout as before. We treat it as a binary of from x,y,z
*/
TEST(TetLinear, testdNdxi)
{
	ShapeFunctions::TetLinear testFunctions;
	const int Nne = 4;

	std::array<std::array<float, 3>, Nne> xi1 = { { {0,0,0},{1,0,0},{0,1,0},{0,0,1} } };

	for (int Ai = 0; Ai < Nne; Ai++)
	{
		Eigen::Vector3f output1 = testFunctions.dNdxi(xi1[Ai], Ai);
		switch (Ai) {
		case 0:
			ASSERT_FLOAT_EQ(output1[0], -1.0f);
			ASSERT_FLOAT_EQ(output1[1], -1.0f);
			ASSERT_FLOAT_EQ(output1[2], -1.0f);
			break;
		case 1:
			ASSERT_FLOAT_EQ(output1[0], 1.0f);
			ASSERT_FLOAT_EQ(output1[1], 0.0f);
			ASSERT_FLOAT_EQ(output1[2], 0.0f);
			break;
		case 2:
			ASSERT_FLOAT_EQ(output1[0], 0.0f);
			ASSERT_FLOAT_EQ(output1[1], 1.0f);
			ASSERT_FLOAT_EQ(output1[2], 0.0f);
			break;
		case 3:
			ASSERT_FLOAT_EQ(output1[0], 0.0f);
			ASSERT_FLOAT_EQ(output1[1], 0.0f);
			ASSERT_FLOAT_EQ(output1[2], 1.0f);
			break;
		}
	}
}

TEST(TetLinear, testMapFaceGPtoXi)
{
	ShapeFunctions::TetLinear testFunctions;
	std::array<float, 2> gp = { 0.25f,0.5f };
	std::array<std::array<float, 3>, 4> truthTable = { { {0.25,0.25f,0.5f}, // 1 - 2 - 3; xi + eta + zeta = 1
											{0.0,0.5f,0.25f},  // 0 - 3 - 2; xi = 
											{0.25,0.0f,0.5f}, // 0 - 1 - 3 ; eta = 0
											{ 0.5f,0.25f,0.0f } // 0 - 2 - 1 ; zeta = 0
											} }; // right
	for (int faceIdx = 0; faceIdx < 4; faceIdx++)
	{
		std::array<float, 3> xi = testFunctions.mapFaceGPtoXi(gp, faceIdx);
		EXPECT_FLOAT_EQ(xi[0], truthTable[faceIdx][0]);
		EXPECT_FLOAT_EQ(xi[1], truthTable[faceIdx][1]);
		EXPECT_FLOAT_EQ(xi[2], truthTable[faceIdx][2]);
	}
}

TEST(TetLinear, testN_face)
{
	ShapeFunctions::TetLinear testFunctions;
	// gauss point for integration: (xi,eta) for a face
	std::array<float, 2> gp1 = { 0.0f,0.0f };
	std::array<float, 2> gp2 = { 1.0f,0.0f };
	std::array<float, 2> gp3 = { 0.0f,1.0f };
	std::array<float, 2> gp4 = { 1/3.0f,1/3.0f };
	int faceIdx = 0;
	std::array<float, 3> point1True = { 1.0f,0.0f,0.0f };
	std::array<float, 3> point2True = { 0.0f,1.0f,0.0f };
	std::array<float, 3> point3True = { 0.0f,0.0f,1.0f };
	std::array<float, 3> point4True = { 1/3.0f,1/3.0f,1/3.0f };

	/*
	* We go through each face and make sure that we are hitting the nodes in the proper order
	* EX: shape function N_0 should be 1 when (xi=0,eta=0) and are at A = 0 for each face and 0 elsewhere
	* EX: shape function N_1 should be 1 when (xi=1,eta=0) and are at A = 1 for each face and 0 elsewhere
	*/
	for (faceIdx = 0; faceIdx < 4; faceIdx++)
	{
		for (int A = 0; A < testFunctions.nFaceNodes; A++)
		{
			//std::cout << "Face: " << faceIdx << " A: " << A << std::endl;
			float output1 = testFunctions.N_face(gp1, A);
			float output2 = testFunctions.N_face(gp2, A);
			float output3 = testFunctions.N_face(gp3, A);
			float output4 = testFunctions.N_face(gp4, A);
			EXPECT_FLOAT_EQ(point1True[A], output1);
			EXPECT_FLOAT_EQ(point2True[A], output2);
			EXPECT_FLOAT_EQ(point3True[A], output3);
			EXPECT_FLOAT_EQ(point4True[A], output4);
		}
	}
}

TEST(TetLinear, testdN_face)
{
	ShapeFunctions::TetLinear testFunctions;
	// gauss point for integration: (xi,eta) for a face
	std::array<float, 2> gp1 = { 0.0f,0.0f };
	std::array<float, 2> gp2 = { 1.0f,0.0f };
	std::array<float, 2> gp3 = { 0.0f,1.0f };
	std::array<float, 2> gp4 = { 1 / 3.0f,1 / 3.0f };
	int faceIdx = 0;

	/*
	* For a generic 2D triangular shape:
	*	  ^ eta       2 
	*	  |           |\
	*     |           | \
	*     *----> xi   0--1
	* A = 0 dNdxi = [-1; -1]
	* A = 1 dNdxi = [ 1;  0]
	* A = 2 dNdxi = [ 0;  1]
	*/
	// Truth table for each gausspoint and node---        Node 0         Node 1        Node 2        
	std::array<std::array<float, 2>, 3> point1True = { {{-1.0f,-1.0f}, {1.0f,0.0f}, {0.0f,1.0f} } };
	std::array<std::array<float, 2>, 3> point2True = { {{-1.0f,-1.0f}, {1.0f,0.0f}, {0.0f,1.0f} } };
	std::array<std::array<float, 2>, 3> point3True = { {{-1.0f,-1.0f}, {1.0f,0.0f}, {0.0f,1.0f} } };
	std::array<std::array<float, 2>, 3> point4True = { {{-1.0f,-1.0f}, {1.0f,0.0f}, {0.0f,1.0f} } };
	

	for (faceIdx = 0; faceIdx < 4; faceIdx++)
	{
		for (int A = 0; A < testFunctions.nFaceNodes; A++)
		{
			//std::cout << "Face: " << faceIdx << " A: " << A << std::endl;
			Eigen::Vector2f output1 = testFunctions.dNdxi_face(gp1, A);
			Eigen::Vector2f output2 = testFunctions.dNdxi_face(gp2, A);
			Eigen::Vector2f output3 = testFunctions.dNdxi_face(gp3, A);
			Eigen::Vector2f output4 = testFunctions.dNdxi_face(gp4, A);
			for (int idx = 0; idx < 2; idx++)
			{
				EXPECT_FLOAT_EQ(point1True[A][idx], output1[idx]);
				EXPECT_FLOAT_EQ(point2True[A][idx], output2[idx]);
				EXPECT_FLOAT_EQ(point3True[A][idx], output3[idx]);
				EXPECT_FLOAT_EQ(point4True[A][idx], output4[idx]);
			}
		}
	}
}