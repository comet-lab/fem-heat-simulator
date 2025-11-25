#include <gtest/gtest.h>
#include "ShapeFunctions/TetQuadratic.hpp"
#include <iostream>
#include <string>

TEST(TetQuadratic, testN)
{
	ShapeFunctions::TetQuadratic testFunctions;
	const int Nne = testFunctions.nNodes; // 10 nodes in tetrahedral elements with quadratic shape functions
	// xi has the coordinates of every node in the mesh
	std::array<std::array<float, 3>, 10> xi = { {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0.5,0,0},{0.5,0.5,0},{0,0.5,0},
		{0,0,0.5},{0.5,0,0.5},{0,0.5,0.5}} };
	
	for (int Ai = 0; Ai < Nne; Ai++)
	{
		for (int i = 0; i < 10; i++)
		{
			float output1 = testFunctions.N(xi[i], Ai);
			if (Ai == i)
				EXPECT_FLOAT_EQ(1.0F, output1);
			else
				EXPECT_FLOAT_EQ(0.0F, output1);
		}
		float output2 = testFunctions.N({ 1 / 4.0f, 1 / 4.0f, 1 / 4.0f }, Ai);
		if (Ai < 4)
			EXPECT_FLOAT_EQ(-1 / 8.0f, output2);
		else
			EXPECT_FLOAT_EQ(1 / 4.0f, output2);
	}
}

/*
* Testing the derivative of the 3D shape function for tetrahedral elements
* Same A layout as before. We treat it as a binary of from x,y,z
*/
TEST(TetQuadratic, testdNdxi)
{
	ShapeFunctions::TetQuadratic testFunctions;
	const int Nne = testFunctions.nNodes;

	std::array<std::array<float, 3>, 10> xi = { {{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0.5,0,0},{0.5,0.5,0},{0,0.5,0},
		{0,0,0.5},{0.5,0,0.5},{0,0.5,0.5}} };
	// true dNdxi for the location {1/4,1/4,1/4} for each node. 
	std::array<Eigen::Vector3f, Nne> output1True = {
	Eigen::Vector3f{0, 0, 0}, // node 0
	Eigen::Vector3f{ 0,  0,  0}, // node 1
	Eigen::Vector3f{ 0,  0,  0}, // node 2
	Eigen::Vector3f{ 0,  0,  0}, // node 3
	Eigen::Vector3f{ 0, -1, -1}, // node 4
	Eigen::Vector3f{ 1,  1,  0}, // node 5
	Eigen::Vector3f{-1,  0, -1}, // node 6
	Eigen::Vector3f{-1,  -1, 0}, // node 7
	Eigen::Vector3f{ 1,  0,  1}, // node 8
	Eigen::Vector3f{ 0,  1,  1} // node 9
	};
	// true dNdxi when we are at the node the shape function belongs to 
	std::array<Eigen::Vector3f, Nne> output2True = {
	Eigen::Vector3f{-3, -3, -3}, // node 0
	Eigen::Vector3f{ 3,  0,  0}, // node 1
	Eigen::Vector3f{ 0,  3,  0}, // node 2
	Eigen::Vector3f{ 0,  0,  3}, // node 3
	Eigen::Vector3f{ 0, -2, -2}, // node 4
	Eigen::Vector3f{ 2,  2,  0}, // node 5
	Eigen::Vector3f{-2,  0, -2}, // node 6
	Eigen::Vector3f{-2,  -2, 0}, // node 7
	Eigen::Vector3f{ 2,  0,  2}, // node 8
	Eigen::Vector3f{ 0,  2,  2} // node 9
	};

	for (int Ai = 0; Ai < Nne; Ai++)
	{
		Eigen::Vector3f output1 = testFunctions.dNdxi({ 1 / 4.0f, 1 / 4.0f, 1 / 4.0f }, Ai);

		EXPECT_FLOAT_EQ(output1True[Ai](0), output1(0));
		EXPECT_FLOAT_EQ(output1True[Ai](1), output1(1));
		EXPECT_FLOAT_EQ(output1True[Ai](2), output1(2));

		Eigen::Vector3f output2 = testFunctions.dNdxi(xi[Ai], Ai);
		EXPECT_FLOAT_EQ(output2True[Ai](0), output2(0));
		EXPECT_FLOAT_EQ(output2True[Ai](1), output2(1));
		EXPECT_FLOAT_EQ(output2True[Ai](2), output2(2));
	}
}

TEST(TetQuadratic, testMapFaceGPtoXi)
{
	ShapeFunctions::TetQuadratic testFunctions;
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

TEST(TetQuadratic, testN_face)
{
	// A face will be a quadratic triangle 

	ShapeFunctions::TetQuadratic testFunctions;
	// gauss point for integration: (xi,eta) for a face
	std::array<std::array<float, 2>, 6> xi = { {{0,0},{1,0},{0,1},{0.5,0},{0.5,0.5},{0,0.5}} };
	std::array<float, 2> gp4 = { 1 / 3.0f,1 / 3.0f };
	int faceIdx = 0;
	std::array<float, 3> point1True = { 1.0f,0.0f,0.0f };
	std::array<float, 3> point2True = { 0.0f,1.0f,0.0f };
	std::array<float, 3> point3True = { 0.0f,0.0f,1.0f };
	std::array<float, 3> point4True = { 1 / 3.0f,1 / 3.0f,1 / 3.0f };

	for (int A = 0; A < testFunctions.nFaceNodes; A++)
	{
		//std::cout << "Face: " << faceIdx << " A: " << A << std::endl;
		for (int i = 0; i < 6; i++)
		{
			float output1 = testFunctions.N_face(xi[i], A);
			if (i == A)
				EXPECT_FLOAT_EQ(1.0f, output1); // 1 at the designated node
			else
				EXPECT_FLOAT_EQ(0.0f, output1); // 0 elsewhere
		}

		float output2 = testFunctions.N_face({ 1 / 3.0f, 1 / 3.0f }, A);
		if (A < 3) // nodes 0,1,2 are vertices
			EXPECT_FLOAT_EQ(-1/9.0f, output2);
		else // nodes 3,4,5 are midpoints
			EXPECT_FLOAT_EQ(4/9.0f, output2);
		
	}
}


TEST(TetQuadratic, testdN_face)
{
	ShapeFunctions::TetQuadratic testFunctions;
	
	/*
	* For a generic 2D triangular shape:
	*	  ^ eta       2
	*	  |           | \
	*     |           5  4
	*     *----> xi   |    \
	*                 0 -3- 1
	* 
	*/
	// gauss point for integration: (xi,eta) for a face
	std::array<std::array<float, 2>, 6> xi = { {{0,0},{1,0},{0,1},{0.5,0},{0.5,0.5},{0,0.5}} };

	// true dNdxi for the location {1/3,1/3} for each node. 
	std::array<Eigen::Vector2f, 6> output1True = {
	Eigen::Vector2f{-1/3.0f, -1 / 3.0f}, // node 0
	Eigen::Vector2f{ 1 / 3.0f,  0.0f}, // node 1
	Eigen::Vector2f{ 0.0f,  1 / 3.0f}, // node 2
	Eigen::Vector2f{ 0.0f, -4/3.0f}, // node 3
	Eigen::Vector2f{ 4/3.0f,  4/3.0f}, // node 4
	Eigen::Vector2f{-4/3.0f,  0.0f}, // node 5
	};

	// true dNdxi when we are at the node the shape function belongs to 
	std::array<Eigen::Vector2f, 6> output2True = {
	Eigen::Vector2f{-3, -3}, // node 0
	Eigen::Vector2f{ 3,  0}, // node 1
	Eigen::Vector2f{ 0,  3}, // node 2
	Eigen::Vector2f{ 0, -2}, // node 3
	Eigen::Vector2f{ 2,  2}, // node 4
	Eigen::Vector2f{-2,  0} // node 5
	};

	for (int A = 0; A < testFunctions.nFaceNodes; A++)
	{
		Eigen::Vector2f output1 = testFunctions.dNdxi_face({ 1 / 3.0f, 1 / 3.0f }, A);
		EXPECT_NEAR(output1True[A](0), output1(0),0.000001);
		EXPECT_NEAR(output1True[A](1), output1(1),0.000001);

		Eigen::Vector2f output2 = testFunctions.dNdxi_face(xi[A], A);
		EXPECT_FLOAT_EQ(output2True[A](0), output2(0));
		EXPECT_FLOAT_EQ(output2True[A](1), output2(1));
	}
}