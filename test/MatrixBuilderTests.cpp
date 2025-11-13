#include <gtest/gtest.h>
#include "MatrixBuilder.h"
#include "ShapeFunctions/HexLinear.hpp"
#include "ShapeFunctions/TetLinear.hpp"
#include <iostream>
#include <string>

class HexBuilder : public testing::Test {
protected:
	std::vector<Node> nodes;
	std::vector<BoundaryFace> boundaryFaces;
	MatrixBuilder<ShapeFunctions::HexLinear>* mb;
	ShapeFunctions::HexLinear testHexLin; 
	Mesh mesh;
	Element elem;
	void SetUp() override {
		nodes.resize(8);
		for (int i = 0; i < 8; i++) {
			nodes[i].x = 0 + i%2;
			nodes[i].y = 0 + (i/2)%2;
			nodes[i].z = 0 + (i/4);
			elem.nodes.push_back(i);
		}
		mesh.setNodes(nodes);
		mesh.setElements({ elem });

		boundaryFaces.resize(6);
		for (int i = 0; i < 6; i++) {
			boundaryFaces[i].elemID = 0;
			boundaryFaces[i].localFaceID = i;
			boundaryFaces[i].nodes.assign(testHexLin.faceConnectivity[i].begin(), testHexLin.faceConnectivity[i].end());
			boundaryFaces[i].type = FLUX;
			boundaryFaces[i].value = 1;
		}
		boundaryFaces[0].type = HEATSINK;
		mesh.setBoundaryFaces(boundaryFaces);

		mb = new MatrixBuilder<ShapeFunctions::HexLinear>(mesh);
	};

	void TearDown() override 
	{
		if (mb) {
			delete mb;
			mb = nullptr;
		}
	};
};

class TetBuilder : public testing::Test {
protected:
	std::vector<Node> nodes;
	std::vector<BoundaryFace> boundaryFaces;
	MatrixBuilder<ShapeFunctions::TetLinear>* mb;
	ShapeFunctions::TetLinear testTetLin;
	Mesh mesh;
	Element elem;
	void SetUp() override {
		nodes.resize(4);
		nodes[0].x = 0; nodes[0].y = 0; nodes[0].z = 0;
		nodes[1].x = 2; nodes[1].y = 0; nodes[1].z = 0;
		nodes[2].x = 0; nodes[2].y = 2; nodes[2].z = 0;
		nodes[3].x = 0; nodes[3].y = 0; nodes[3].z = 2;
		for (int i = 0; i < 8; i++) {
			elem.nodes.push_back(i);
		}
		mesh.setNodes(nodes);
		mesh.setElements({ elem });

		boundaryFaces.resize(4);
		for (int i = 0; i < 4; i++) {
			boundaryFaces[i].elemID = 0;
			boundaryFaces[i].localFaceID = i;
			boundaryFaces[i].nodes.assign(testTetLin.faceConnectivity[i].begin(), testTetLin.faceConnectivity[i].end());
			boundaryFaces[i].type = FLUX;
			boundaryFaces[i].value = 1;
		}
		boundaryFaces[0].type = HEATSINK;
		mesh.setBoundaryFaces(boundaryFaces);

		mb = new MatrixBuilder<ShapeFunctions::TetLinear>(mesh);
	};

	void TearDown() override
	{
		if (mb) {
			delete mb;
			mb = nullptr;
		}
	};
};

TEST_F(HexBuilder, testSetNodeList)
{
	for (int i = 0; i < 8; i++)
	{
		ASSERT_EQ(nodes[i].x, mesh.nodes()[i].x);
		ASSERT_EQ(nodes[i].y, mesh.nodes()[i].y);
		ASSERT_EQ(nodes[i].z, mesh.nodes()[i].z);
	}
}


/**
* Tests the 3D tri-linear shape function for hexahedral elements.
* We are enforcing the order here as well that 
* A[0] = (0,0,0)
* A[1] = (1,0,0)
* A[2] = (0,1,0)
* A[3] = (1,1,0)
* A[4] = (0,0,1) ... etc.
*/
TEST_F(HexBuilder, testCalculateHexFunction3D)
{
	int Nne = 8; // 8 nodes in hexahedral elements with linear shape functions
	// xi1 has the coordinates at the nodal locations in the element
	std::array<std::array<float, 3>, 8> xi1 = { { {-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1},
												  {-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1} } };
	// xi2 has the coordinates of a different node
	std::array<std::array<float, 3>, 8> xi2 = { { {1,1,1},{-1,1,1},{1,-1,1},{-1,-1,1},
												  {1,1,-1},{-1,1,-1},{1,-1,-1},{-1,-1,-1} } };
	for (int Ai = 0; Ai < Nne; Ai++) {
		float output1 = testHexLin.N(xi1[Ai], Ai);
		float output2 = testHexLin.N(xi2[Ai], Ai);
		float output3 = testHexLin.N({ 0,0,0 }, Ai);

		EXPECT_FLOAT_EQ(1.0f, output1);
		EXPECT_FLOAT_EQ(0.0f, output2);
		EXPECT_FLOAT_EQ(1/8.0f, output3);
	}
}

/*
* Testing the derivative of the 3D shape function for hexahedral elements
* Same A layout as before. We treat it as a binary of from x,y,z
*/
TEST_F(HexBuilder, testCalculateHexFunctionDeriv3D)
{
	int Nn1d = 2;
	int Nne = 8;
	std::array<std::array<float, 3>, 8> truthTable = { { {-1/2.0f,-1 / 2.0f,-1 / 2.0f}, {1 / 2.0f,-1 / 2.0f,-1 / 2.0f}, 
														 {-1 / 2.0f,1 / 2.0f,-1 / 2.0f},{1 / 2.0f,1 / 2.0f,-1 / 2.0f},
														 {-1 / 2.0f,-1 / 2.0f,1 / 2.0f},{1 / 2.0f,-1 / 2.0f,1 / 2.0f},
														 {-1 / 2.0f,1 / 2.0f,1 / 2.0f},{1 / 2.0f,1 / 2.0f,1 / 2.0f} } };

	std::array<std::array<float, 3>, 8> xi1 = { { {-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1},
												  {-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1} } };

	for (int Ai = 0; Ai < Nne; Ai++) 
	{	
		Eigen::Vector3f output1 = testHexLin.dNdxi(xi1[Ai], Ai);
		Eigen::Vector3f output2 = testHexLin.dNdxi({0.0f,0.0f,0.0f}, Ai);

		for (int i = 0; i < 3; i++) 
		{
			EXPECT_FLOAT_EQ(truthTable[Ai][i], output1(i)) << "Ai: " << Ai << " i: " << i;
			EXPECT_FLOAT_EQ(truthTable[Ai][i]/4.0f, output2(i)) << "Ai: " << Ai << " i: " << i;
		}
	}
}

/*
* Calculating the Jacobian. Based on the element we created, the Jacobian should simply be
* a diagonal element with 0.5 for each diagonal element
*/
TEST_F(HexBuilder, testCalculateJacobian)
{
	std::vector<Eigen::Vector3f> dNdxi(8);
	for (int a = 0; a < 8; a++)
	{
		dNdxi[a] = ShapeFunctions::HexLinear::dNdxi({ 1.0f,-1.0f,0.0f }, a);
	}
	
	Eigen::Matrix3f J = mb->calculateJ(elem, dNdxi); // position (xi) doesn't matter in this test case

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
		{
			if (i == j)
			{
				// diagonal should be 1
				EXPECT_FLOAT_EQ(J(i, j), 0.5f);
			}
			else
			{
				// off diagonal should be 0
				EXPECT_FLOAT_EQ(J(i, j), 0);
			}
		}
	}
}

/*
* Testing the calculation of Me. For the single element we have it should be straightforward
* detJ should be 1/8. Diagonal elements should be 1/8 * ( 8/27). 
*/
TEST_F(HexBuilder, testCalculateHexLinMe)
{

	Eigen::Matrix<float, 8, 8> Me = mb->calculateMe(elem);

	// The truth values were calculated in matlab assuming deltaX = deltaY = deltaZ = 0.5
	float scale[8] = { 1.0f, 2.0f, 2.0f, 4.0f, 2.0f, 4.0f, 4.0f, 8.0f };
	for (int Ai = 0; Ai < 8; Ai++) 
	{
		EXPECT_FLOAT_EQ(1/27.0f,Me(Ai, Ai));
		EXPECT_FLOAT_EQ((1 / (scale[Ai]*27.0f)), Me(Ai, 0));
	}
	
}

/*
* Testing the calculation of Ke. For the single element we have it should be straightforward
*
*/
TEST_F(HexBuilder, testCalculateHexLinKe)
{

	Eigen::Matrix<float, 8, 8> Ke = mb->calculateKe(elem);

	// The truth values were calculated in matlab assuming deltaX = deltaY = deltaZ = 0.5
	float scale[8] = { 1.0f, 0.0f, 0.0f, -1/4.0f, 0.0f, -1/4.0f, -1/4.0f, -1/4.0f };
	float tolerance = 0.000001;
	for (int Ai = 0; Ai < 8; Ai++)
	{
		EXPECT_FLOAT_EQ(1 / 3.0f, Ke(Ai, Ai));
		EXPECT_NEAR( (scale[Ai] / 3.0f), Ke(Ai, 0),tolerance);
	}

}

/*
* Testing the calculation of Feflux for single HexLinear Element
*
*/
TEST_F(HexBuilder, testCalculateFeFlux)
{

	//Face 0: Bottom Face
	Eigen::Vector<float, 8> FeFlux = mb->calculateFeFlux(elem,0,1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 4) || (A == 5) || (A == 6) || (A == 7))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}

	//Face 1: Top Face
	FeFlux = mb->calculateFeFlux(elem, 1, 1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 0) || (A == 1) || (A == 2) || (A == 3))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}

	//Face 2: Front Face
	FeFlux = mb->calculateFeFlux(elem, 2, 1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 3) || (A == 2) || (A == 6) || (A == 7))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}

	//Face 3: Back Face
	FeFlux = mb->calculateFeFlux(elem, 3, 1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 0) || (A == 1) || (A == 4) || (A == 5))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}

	//Face 4: Left Face
	FeFlux = mb->calculateFeFlux(elem, 4, 1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 1) || (A == 3) || (A == 5) || (A == 7))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}

	//Face 5: Right Face
	FeFlux = mb->calculateFeFlux(elem, 5, 1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 0) || (A == 2) || (A == 4) || (A == 6))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}
}

/*
* Testing the calculation of Feflux. For the single element we have it should be straightforward
*
*/
TEST_F(HexBuilder, testCalculateFeConv)
{
	constexpr int faceNodes[6][4] = {
		{0,1,2,3}, {4,5,6,7}, {0,1,4,5},
		{2,3,6,7}, {0,2,4,6}, {1,3,5,7}
	};
	std::array<std::array<float, 4>, 4> scale = { {	{ 1, 1 / 2.0f, 1 / 2.0f, 1 / 4.0f},
													{ 1 / 2.0f, 1, 1 / 4.0f, 1 / 2.0f},
													{ 1 / 2.0f, 1 / 4.0f, 1, 1 / 2.0f},
													{ 1 / 4.0f, 1 / 2.0f, 1 / 2.0f, 1} } };
	int A = 0;
	int B = 0;
	for (int f = 0; f < 6; f++)
	{
		Eigen::MatrixXf FeConv = mb->calculateFeConv(elem, f);
		// all of these nodes should be scaled relative to scale matrix
		const int* nodes = faceNodes[f];
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				A = nodes[i];
				B = nodes[j];
				EXPECT_FLOAT_EQ(FeConv(A, B), scale[i][j] / 9.0f);
			}
		}
		// All of these nodes should be on the opposite face so have nodes
		// that should have a 0 value in the matrix
		if (f % 2)
			f = (f - 1);
		else
			f = (f + 1);
		nodes = faceNodes[f];
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				A = nodes[i];
				B = nodes[j];
				EXPECT_FLOAT_EQ(FeConv(A, B), 0);
			}
		}
	}
}

TEST_F(HexBuilder, testSetNodeMap)
{
	// only bottom face is dirichlet
	std::vector<long> trueMap = { -1 ,-1,-1,-1, 0, 1, 2, 3 };
	std::vector<long> trueValid = { 4,5,6,7 };
	for (int i = 0; i < 8; i++)
	{
		EXPECT_EQ(trueMap[i], mb->nodeMap()[i]);
	}
	for (int i = 0; i < 4; i++)
	{
		EXPECT_EQ(trueValid[i], mb->validNodes()[i]);
	}
	EXPECT_EQ(trueValid.size(), mb->validNodes().size());
}


/*
* Testing the construction of the shape functions for Linear Tetrahedral elements
*/
TEST_F(TetBuilder, testCalculateN)
{
	const int Nne = 4; // 8 nodes in tetrahedral elements with linear shape functions
	// xi1 has the coordinates at the nodal locations in the element
	std::array<std::array<float, 3>, Nne> xi1 = { { {0,0,0},{1,0,0},{0,1,0},{0,0,1} } };
	// xi2 has the coordinates of a different node
	std::array<std::array<float, 3>, Nne> xi2 = { { {1,0,0},{0,1,0},{0,0,1},{0,0,0} } };
	for (int Ai = 0; Ai < Nne; Ai++) {
		float output1 = testTetLin.N(xi1[Ai], Ai);
		float output2 = testTetLin.N(xi2[Ai], Ai);
		float output3 = testTetLin.N({ 1/3.0f, 1 / 3.0f, 1 / 3.0f }, Ai);

		EXPECT_FLOAT_EQ(1.0f, output1);
		EXPECT_FLOAT_EQ(0.0f, output2);
		if (Ai == 0)
			EXPECT_NEAR(0.0f, output3,0.000001);
		else
			EXPECT_FLOAT_EQ(1 / 3.0f, output3);
	}
}

/* Testing the construction of the spatial derivatives of the shape functions
* for linear tetrahedral elements
*/
TEST_F(TetBuilder, testCalculatedNdxi)
{
	const int Nne = 4;

	std::array<std::array<float, 3>, Nne> xi1 = { { {0,0,0},{1,0,0},{0,1,0},{0,0,1} } };

	for (int Ai = 0; Ai < Nne; Ai++)
	{
		Eigen::Vector3f output1 = testTetLin.dNdxi(xi1[Ai], Ai);
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

/* Testing Calculation of the jacobain when using linear tetrahedral elements
* Element we are using has Jacobian = diag(2*ones(3,1))
*/
TEST_F(TetBuilder, testCalculateJ)
{
	const int Nne = ShapeFunctions::TetLinear::nNodes;
	std::vector<Eigen::Vector3f> dNdxi(Nne);
	for (int a = 0; a < Nne; a++)
	{
		dNdxi[a] = ShapeFunctions::TetLinear::dNdxi({ 1.0f,-1.0f,0.0f }, a);
	}

	Eigen::Matrix3f J = mb->calculateJ(elem, dNdxi); // position (xi) doesn't matter in this test case

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++)
		{
			if (i == j)
			{
				// diagonal should be 2.0
				EXPECT_FLOAT_EQ(J(i, j), 2.0f) << "i: " << i << " j: " << j << std::endl;
			}
			else
			{
				// off diagonal should be 0
				EXPECT_FLOAT_EQ(J(i, j), 0) << "i: " << i << " j: " << j << std::endl;
			}
		}
	}
}

/* Testing Calculation of the thermal mass matrix when using linear tetrahedral elements
* Element we are using has Jacobian = diag(2*ones(3,1))
*/
TEST_F(TetBuilder, testCalculateMe)
{
	const int Nne = testTetLin.nNodes;
	Eigen::Matrix<float, Nne, Nne> Me = mb->calculateMe(elem);

	// The truth values were calculated in matlab assuming deltaX = deltaY = deltaZ = 2.0

	float scale[Nne] = { 1.0f, 2.0f, 2.0f, 2.0f };
	for (int Ai = 0; Ai < Nne; Ai++)
	{
		EXPECT_FLOAT_EQ(8 / 60.0f, Me(Ai, Ai));
		EXPECT_FLOAT_EQ((8 / (scale[Ai] * 60.0f)), Me(Ai, 0));
	}

}

/* Testing Calculation of the stiffness matrix when using linear tetrahedral elements
* Element we are using has Jacobian = diag(2*ones(3,1))
*/
TEST_F(TetBuilder, testCalculateKe)
{
	const int Nne = testTetLin.nNodes;
	Eigen::Matrix<float, Nne, Nne> Ke = mb->calculateKe(elem);

	Eigen::Matrix<float, Nne, Nne> trueKe;
	trueKe << 1.0, -1 / 3.0f, -1 / 3.0f, -1 / 3.0f,
		-1 / 3.0f, 1 / 3.0f, 0.0f, 0.0f,
		-1 / 3.0f, 0.0f, 1 / 3.0f, 0.0f,
		-1 / 3.0f, 0.0f, 0.0f, 1 / 3.f;
	float scale[Nne] = { 1.0f, 0.0f, 0.0f, -1 / 4.0f};
	float tolerance = 0.000001;
	for (int Ai = 0; Ai < Nne; Ai++)
	{
		for (int Bi = 0; Bi < Nne; Bi++)
		{
			EXPECT_FLOAT_EQ(trueKe(Ai, Bi), Ke(Ai, Bi));
		}
	}
}

/*
* Testing construction of FeFlux vector for linear tetrahedral elements
*/
TEST_F(TetBuilder, testCalculateFeFlux)
{
	const int nNe = testTetLin.nNodes;
	//Face 0: Bottom Face
	Eigen::Vector<float, nNe> FeFlux = mb->calculateFeFlux(elem, 0, 1);
	for (int A = 0; A < nNe; A++)
	{
		if (A == 3)
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 2/3.0f);
	}

	//Face 1:
	FeFlux = mb->calculateFeFlux(elem, 1, 1);
	for (int A = 0; A < nNe; A++)
	{
		if (A == 2)
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 2 / 3.0f);
	}

	//Face 2: is slanted and has a larger area
	FeFlux = mb->calculateFeFlux(elem, 2, 1);
	for (int A = 0; A < nNe; A++)
	{
		if (A == 0)
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), sqrt(3)*2 / 3.0f); // area is larger by factor of sqrt(3) so flux is larger by that amount
	}

	//Face 3:
	FeFlux = mb->calculateFeFlux(elem, 3, 1);
	for (int A = 0; A < nNe; A++)
	{
		if (A==1)
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 2 / 3.0f);
	}

}

TEST_F(TetBuilder, testCalculateFeConv)
{
	const int nNe = testTetLin.nNodes;
	const int nFn = testTetLin.nFaceNodes;
	std::array<std::array<float, nFn>, nFn> scale = { {	{ 1.0f, 0.5f, 0.5f},
													{ 0.5f, 1.0f, 0.5f},
													{ 0.5f, 0.5f, 1.0f} } };

	std::array<int, 4> zeroNodes = { 3,2,0,1 }; // node for face = f where Conv value should be 0
	int A = 0;
	int B = 0;
	for (int f = 0; f < 4; f++)
	{
		Eigen::MatrixXf FeConv = mb->calculateFeConv(elem, f);
		// all of these nodes should be scaled relative to scale matrix
		std::array<int, nFn> nodes = testTetLin.faceConnectivity[f];
		for (int i = 0; i < nFn; i++)
		{
			for (int j = 0; j < nFn; j++)
			{
				A = nodes[i];
				B = nodes[j];
				if (f == 2)
					// when f == 2 we have the slanted face which has a larger jacobian determinant by a factor of sqrt(3)
					EXPECT_FLOAT_EQ(FeConv(A, B), sqrt(3)*scale[i][j] / 3.0f); 
				else
					EXPECT_FLOAT_EQ(FeConv(A, B), scale[i][j] / 3.0f);
			}
		}
		// All of these nodes should be on the opposite face so have nodes
		// that should have a 0 value in the matrix
		for (int j = 0; j < nNe; j++)
		{
			EXPECT_FLOAT_EQ(FeConv(zeroNodes[f], j), 0);
			EXPECT_FLOAT_EQ(FeConv(j, zeroNodes[f]), 0);
		}
	}
}
