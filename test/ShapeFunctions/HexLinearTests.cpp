#include <gtest/gtest.h>
#include "ShapeFunctions/HexLinear.hpp"
#include <iostream>
#include <string>


/**
* Tests the 3D tri-linear shape function for hexahedral elements.
* We are enforcing the order here as well that
* A[0] = (0,0,0)
* A[1] = (1,0,0)
* A[2] = (1,1,0)
* A[3] = (0,1,0)
* A[4] = (0,0,1) ... etc.
*/
TEST(HexLinear, testN)
{
	ShapeFunctions::HexLinear testFunctions; 
	int Nne = 8; // 8 nodes in hexahedral elements with linear shape functions
	// xi1 has the coordinates at the nodal locations in the element
	std::array<std::array<float, 3>, 8> xi1 = { { {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
												  {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1} } };
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
														 {1 / 2.0f,1 / 2.0f,-1 / 2.0f},{-1 / 2.0f,1 / 2.0f,-1 / 2.0f},
														 {-1 / 2.0f,-1 / 2.0f,1 / 2.0f},{1 / 2.0f,-1 / 2.0f,1 / 2.0f},
														 {1 / 2.0f,1 / 2.0f,1 / 2.0f},{-1 / 2.0f,1 / 2.0f,1 / 2.0f} } };

	std::array<std::array<float, 3>, 8> xi1 = { { {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
												  {-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1} } };

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

TEST(HexLinear, testMapFaceGPtoXi)
{
	ShapeFunctions::HexLinear testFunctions;
	std::array<float, 2> gp = { -1.0f,1.0f };
	std::array<std::array<float,3>, 6> truthTable = { { {1.0f,-1.0f,-1.0f}, // bot
											{-1.0,1.0f,1.0f}, // top
											{-1.0,-1.0f,1.0f}, // back
											{1.0,1.0f,-1.0f}, // front
											{-1.0,1.0f,-1.0f}, // left
											{1.0,-1.0f,1.0f}} }; // right
	for (int faceIdx = 0; faceIdx < 6; faceIdx++)
	{
		std::array<float, 3> xi = testFunctions.mapFaceGPtoXi(gp, faceIdx);
		EXPECT_FLOAT_EQ(xi[0], truthTable[faceIdx][0]);
		EXPECT_FLOAT_EQ(xi[1], truthTable[faceIdx][1]);
		EXPECT_FLOAT_EQ(xi[2], truthTable[faceIdx][2]);
	}
}

TEST(HexLinear, testN_face)
{
	ShapeFunctions::HexLinear testFunctions;
	// gauss point for integration: (xi,eta) for a face
	std::array<float, 2> gp1 = { 0.0f,0.0f };
	std::array<float, 2> gp2 = { -1.0f,-1.0f };
	std::array<float, 2> gp3 = { 1.0f,-1.0f };
	std::array<float, 2> gp4 = { 1.0f,1.0f };
	std::array<float, 2> gp5 = { -1.0f,1.0f };
	int faceIdx = 0;
	std::array<float, 4> point2True = { 1.0f,0.0f,0.0f,0.0f };
	std::array<float, 4> point3True = { 0.0f,1.0f,0.0f,0.0f };
	std::array<float, 4> point4True = { 0.0f,0.0f,1.0f,0.0f };
	std::array<float, 4> point5True = { 0.0f,0.0f,0.0f,1.0f };

	/* 
	* We go through each face and make sure that we are hitting the nodes in the proper order
	* EX: shape function N_0 should be 1 when (xi=-1,eta=-1) and are at A = 0 for each face and 0 elsewhere
	* EX: shape function N_1 should be 1 when (xi=1,eta=-1) and are at A = 1 for each face and 0 elsewhere
	*/
	for (faceIdx = 0; faceIdx < 6; faceIdx++)
	{
		for (int A = 0; A < testFunctions.nFaceNodes; A++)
		{
			std::cout << "Face: " << faceIdx << " A: " << A << std::endl;
			float output = testFunctions.N_face(gp1, A, faceIdx);
			float output2 = testFunctions.N_face(gp2, A, faceIdx);
			float output3 = testFunctions.N_face(gp3, A, faceIdx);
			float output4 = testFunctions.N_face(gp4, A, faceIdx);
			float output5 = testFunctions.N_face(gp5, A, faceIdx);
			EXPECT_FLOAT_EQ(1 / 4.0f, output);
			EXPECT_FLOAT_EQ(point2True[A], output2);
			EXPECT_FLOAT_EQ(point3True[A], output3);
			EXPECT_FLOAT_EQ(point4True[A], output4);
			EXPECT_FLOAT_EQ(point5True[A], output5);
		}
	}
}

TEST(HexLinear, testdN_face)
{
	ShapeFunctions::HexLinear testFunctions;
	// gauss point for integration: (xi,eta) for a face
	std::array<float, 2> gp1 = { -1.0f,-1.0f };
	std::array<float, 2> gp2 = { 1.0f,-1.0f };
	std::array<float, 2> gp3 = { 1.0f,1.0f };
	std::array<float, 2> gp4 = { -1.0f,1.0f };
	int faceIdx = 0;

	/*
	* For a generic 2D rectanguglar shape:
	*	  ^ eta       3 ----- 2
	*	  |           |       |
	*     |           |       |
	*     *----> xi   0 ----- 1
	* A = 0 dNdxi = 1/4*[- (1 - eta); - (1 - xi)]
	* A = 1 dNdxi = 1/4*[  (1 - eta); - (1 + xi)]
	* A = 2 dNdxi = 1/4*[  (1 + eta);   (1 + xi)]
	* A = 3 dNdxi = 1/4*[- (1 + eta);   (1 - xi)]
	*/
	// Truth table for each gausspoint and node---        Node 0         Node 1        Node 2         Node 3
	std::array<std::array<float, 2>, 4> point1True = { {{-0.5f,-0.5f}, {0.5f,0.0f},  {0.0f,0.0f},  {0.0f,0.5f} } };
	std::array<std::array<float, 2>, 4> point2True = { {{-0.5f,0.0f},  {0.5f,-0.5f}, {0.0f,0.5f},  {0.0f,0.0f} } };
	std::array<std::array<float, 2>, 4> point3True = { {{0.0f,0.0f},   {0.0f,-0.5f}, {0.5f,0.5f} , {-0.5f,0.0f} } };
	std::array<std::array<float, 2>, 4> point4True = { {{0.0f,-0.5f},  {0.0f,0.0f},  {0.5f,0.0f} , {-0.5f,0.5f} } };

	for (faceIdx = 0; faceIdx < 6; faceIdx++)
	{
		for (int A = 0; A < testFunctions.nFaceNodes; A++)
		{
			std::cout << "Face: " << faceIdx << " A: " << A << std::endl;
			Eigen::Vector2f output1 = testFunctions.dNdxi_face(gp1, A, faceIdx);
			Eigen::Vector2f output2 = testFunctions.dNdxi_face(gp2, A, faceIdx);
			Eigen::Vector2f output3 = testFunctions.dNdxi_face(gp3, A, faceIdx);
			Eigen::Vector2f output4 = testFunctions.dNdxi_face(gp4, A, faceIdx);
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