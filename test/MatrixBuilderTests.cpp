#include <gtest/gtest.h>
#include "MatrixBuilder.h"
#include <iostream>
#include <string>

class BaseClass : public testing::Test {
protected:
	std::vector<Node> nodes;
	std::vector<BoundaryFace> boundaryFaces;
	MatrixBuilder* mb;
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
			boundaryFaces[i].nodes.assign(FaceConnectivity::HEX8[i].begin(), FaceConnectivity::HEX8[i].end());
			boundaryFaces[i].type = FLUX;
			boundaryFaces[i].value = 1;
		}
		mesh.setBoundaryFaces(boundaryFaces);

		mb = new MatrixBuilder(mesh);
	};

	void TearDown() override 
	{
		if (mb) {
			delete mb;
			mb = nullptr;
		}
	};
};

TEST_F(BaseClass, testSetNodeList)
{
	for (int i = 0; i < 8; i++)
	{
		ASSERT_EQ(nodes[i].x, mesh.nodes()[i].x);
		ASSERT_EQ(nodes[i].y, mesh.nodes()[i].y);
		ASSERT_EQ(nodes[i].z, mesh.nodes()[i].z);
	}
}

/**
* Tests the 1D shape function for a linear element where xi \in [-1 1] 
* This function would not work for tetrahedral elements where usually xi is between 0 and 1
*/
TEST_F(BaseClass, testCalculateHexFunction1D)
{
	std::array<std::array<float, 2>, 3> truthVal = { { {1, 0}, {0.5,0.5}, {0,1} } };
	std::array<float, 3> xi = { -1,0,1 };
	std::array<float, 2> A = { 0,1 };
	for (int i = 0; i < 3; i++) 
	{
		for (int j = 0; j < 2; j++)
		{
			EXPECT_EQ(mb->calculateHexFunction1D(xi[i],A[j]), truthVal[i][j]) << "xi[i]: " << xi[i] << " A[j]: " << A[j];
		}
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
TEST_F(BaseClass, testCalculateHexFunction3D)
{
	int Nne = 8; // 8 nodes in hexahedral elements with linear shape functions
	// xi1 has the coordinates at the nodal locations in the element
	std::array<std::array<float, 3>, 8> xi1 = { { {-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1},
												  {-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1} } };
	// xi2 has the coordinates of a different node
	std::array<std::array<float, 3>, 8> xi2 = { { {1,1,1},{-1,1,1},{1,-1,1},{-1,-1,1},
												  {1,1,-1},{-1,1,-1},{1,-1,-1},{-1,-1,-1} } };
	for (int Ai = 0; Ai < Nne; Ai++) {
		float output1 = mb->calculateHexFunction3D(xi1[Ai], Ai);
		float output2 = mb->calculateHexFunction3D(xi2[Ai], Ai);
		float output3 = mb->calculateHexFunction3D({ 0,0,0 }, Ai);

		EXPECT_FLOAT_EQ(1.0f, output1);
		EXPECT_FLOAT_EQ(0.0f, output2);
		EXPECT_FLOAT_EQ(1/8.0f, output3);
	}
}

/*
* Testing the derivative of the 1D shape function for hexahedral elements
*/
TEST_F(BaseClass, testCalculateHexFunctionDeriv1D)
{
	for (int Ai = 0; Ai < 2; Ai++) {

		float output1 = mb->calculateHexFunctionDeriv1D(-1, Ai);
		float output2 = mb->calculateHexFunctionDeriv1D(0, Ai);
		float output3 = mb->calculateHexFunctionDeriv1D(1, Ai);
		float output4 = mb->calculateHexFunctionDeriv1D(-0.5, Ai);

		if (Ai == 0) {
			EXPECT_FLOAT_EQ(-1 / 2.0f, output1);
			EXPECT_FLOAT_EQ(-1 / 2.0f, output2);
			EXPECT_FLOAT_EQ(-1 / 2.0f, output3);
			EXPECT_FLOAT_EQ(-1 / 2.0f, output4);
		}
		else if (Ai == 1) {
			EXPECT_FLOAT_EQ(1 / 2.0f, output1);
			EXPECT_FLOAT_EQ(1 / 2.0f, output2);
			EXPECT_FLOAT_EQ(1 / 2.0f, output3);
			EXPECT_FLOAT_EQ(1 / 2.0f, output4);
		}
	}
}

/*
* Testing the derivative of the 3D shape function for hexahedral elements
* Same A layout as before. We treat it as a binary of from x,y,z
*/
TEST_F(BaseClass, testCalculateHexFunctionDeriv3D)
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
		Eigen::Vector3f output1 = mb->calculateHexFunctionDeriv3D(xi1[Ai], Ai);
		Eigen::Vector3f output2 = mb->calculateHexFunctionDeriv3D({0.0f,0.0f,0.0f}, Ai);

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
TEST_F(BaseClass, testCalculateJacobian)
{
	Eigen::Matrix3f J = mb->calculateJ(elem, { 1.0f,-1.0f,0.0f }); // position (xi) doesn't matter in this test case

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
TEST_F(BaseClass, testCalculateHexLinMe)
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
TEST_F(BaseClass, testCalculateHexLinKe)
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
* Testing the calculation of Feflux. For the single element we have it should be straightforward
*
*/
TEST_F(BaseClass, testCalculateFeFlux)
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
TEST_F(BaseClass, testCalculateFeConv)
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

TEST_F(BaseClass, TestInd2Sub1)
{
	std::array<long,3> sub;
	long index = 0;
	std::array<long,3> size = { 10,10,10 };
	sub = mb->ind2sub(index, size);
	EXPECT_FLOAT_EQ(0, sub[0]);
	EXPECT_FLOAT_EQ(0, sub[1]);
	EXPECT_FLOAT_EQ(0, sub[2]);
}

TEST_F(BaseClass, TestInd2Sub2)
{
	std::array<long, 3> sub;
	long index = 10;
	std::array<long, 3> size = { 10,10,10 };
	sub = mb->ind2sub(index, size);
	EXPECT_FLOAT_EQ(0, sub[0]);
	EXPECT_FLOAT_EQ(1, sub[1]);
	EXPECT_FLOAT_EQ(0, sub[2]);
}

TEST_F(BaseClass, TestInd2Sub3)
{
	std::array<long, 3> sub;
	long index = 100;
	std::array<long, 3> size = { 10,10,10 };
	sub = mb->ind2sub(index, size);
	EXPECT_FLOAT_EQ(0, sub[0]);
	EXPECT_FLOAT_EQ(0, sub[1]);
	EXPECT_FLOAT_EQ(1, sub[2]);
}

TEST_F(BaseClass, TestInd2Sub4)
{
	std::array<long, 3> sub;
	long index = 521;
	std::array<long, 3> size = { 10,10,10 };
	sub = mb->ind2sub(index, size);
	EXPECT_FLOAT_EQ(1, sub[0]);
	EXPECT_FLOAT_EQ(2, sub[1]);
	EXPECT_FLOAT_EQ(5, sub[2]);
}
