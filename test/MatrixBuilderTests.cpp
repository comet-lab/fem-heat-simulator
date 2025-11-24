#include <gtest/gtest.h>
#include "MatrixBuilder.hpp"
#include <iostream>
#include <string>

class HexBuilder : public testing::Test {
protected:
	MatrixBuilder matrixBuilder;
	Mesh mesh;
	void SetUp() override {
		std::vector<float> xPos = { -1,0,1 };
		std::vector<float> yPos = { -1,0,1 };
		std::vector<float> zPos = { -2,-1,0 };
		std::array<BoundaryType, 6> bc = { FLUX,HEATSINK,FLUX,CONVECTION,CONVECTION,CONVECTION };
		mesh = Mesh::buildCubeMesh(xPos,yPos,zPos,bc);

		matrixBuilder.setMesh(mesh);
		matrixBuilder.precomputeShapeFunctions<ShapeFunctions::HexLinear>();
	};

	void TearDown() override 
	{

	};
};

class TetBuilder : public testing::Test {
protected:
	std::vector<Node> nodes;
	std::vector<BoundaryFace> boundaryFaces;
	MatrixBuilder matrixBuilder;
	ShapeFunctions::TetLinear testTetLin;
	Mesh mesh;
	Element elem;
	void SetUp() override {
		nodes.resize(4);
		nodes[0].x = 0; nodes[0].y = 0; nodes[0].z = 0;
		nodes[1].x = 2; nodes[1].y = 0; nodes[1].z = 0;
		nodes[2].x = 0; nodes[2].y = 2; nodes[2].z = 0;
		nodes[3].x = 0; nodes[3].y = 0; nodes[3].z = 2;
		for (int i = 0; i < 4; i++) {
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

		matrixBuilder.setMesh(mesh);
		matrixBuilder.precomputeShapeFunctions<ShapeFunctions::TetLinear>();
	};

	void TearDown() override
	{
	};
};

/*
* Calculating the Jacobian. Based on the element we created, the Jacobian should simply be
* a diagonal element with 1 for each diagonal element
*/
TEST_F(HexBuilder, testCalculateJacobian)
{
	Eigen::Matrix<float,3,8> dNdxi;
	for (int a = 0; a < 8; a++)
	{
		dNdxi.col(a) = ShapeFunctions::HexLinear::dNdxi({1.0f,-1.0f,0.0f}, a);
	}
	
	Eigen::Matrix3f J = matrixBuilder.calculateJ<ShapeFunctions::HexLinear>(mesh.elements()[0], dNdxi); // position (xi) doesn't matter in this test case

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
	matrixBuilder.precomputeElemJ<ShapeFunctions::HexLinear>(mesh.elements()[0]);
	Eigen::Matrix<float, 8, 8> Me = matrixBuilder.calculateIntNaNb<ShapeFunctions::HexLinear>(mesh.elements()[0]);

	// The truth values were calculated in matlab assuming deltaX = deltaY = deltaZ = 0.5
	float scale[8] = { 1.0f, 2.0f, 4.0f, 2.0f, 2.0f, 4.0f, 8.0f, 4.0f };
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
	matrixBuilder.precomputeElemJ<ShapeFunctions::HexLinear>(mesh.elements()[0]);
	Eigen::Matrix<float, 8, 8> Ke = matrixBuilder.calculateIntdNadNb<ShapeFunctions::HexLinear>(mesh.elements()[0]);

	// The truth values were calculated in matlab assuming deltaX = deltaY = deltaZ = 0.5
	float scale[8] = { 1.0f, 0.0f, -1/4.0f, 0.0f, 0.0f, -1/4.0f, -1/4.0f, -1/4.0f };
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
	Eigen::Vector<float, 8> FeFlux = matrixBuilder.calculateFaceIntNa<ShapeFunctions::HexLinear>(mesh.elements()[0],0,1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 4) || (A == 5) || (A == 6) || (A == 7))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}

	//Face 1: Top Face
	FeFlux = matrixBuilder.calculateFaceIntNa<ShapeFunctions::HexLinear>(mesh.elements()[0], 1, 1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 0) || (A == 1) || (A == 2) || (A == 3))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the bottom nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}

	//Face 2: back Face
	FeFlux = matrixBuilder.calculateFaceIntNa<ShapeFunctions::HexLinear>(mesh.elements()[0], 2, 1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 3) || (A == 2) || (A == 6) || (A == 7))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the front  nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}

	//Face 3: front Face
	FeFlux = matrixBuilder.calculateFaceIntNa<ShapeFunctions::HexLinear>(mesh.elements()[0], 3, 1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 0) || (A == 1) || (A == 4) || (A == 5))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the back nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}

	//Face 4: Left Face
	FeFlux = matrixBuilder.calculateFaceIntNa<ShapeFunctions::HexLinear>(mesh.elements()[0], 4, 1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 1) || (A == 2) || (A == 5) || (A == 6))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}

	//Face 5: Right Face
	FeFlux = matrixBuilder.calculateFaceIntNa<ShapeFunctions::HexLinear>(mesh.elements()[0], 5, 1);
	for (int A = 0; A < 8; A++)
	{
		if ((A == 0) || (A == 3) || (A == 4) || (A == 7))
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 0.25);
	}
}

/*
* Testing the calculation of Feflux. For the single element we have it should be straightforward

*/
TEST_F(HexBuilder, testCalculateFeConv)
{
	// Reminder that for the element[0], the determinant of the Jacobian is 1 / 8 which influences
	// the scaling values here. 
	std::array<std::array<float, 4>, 4> scale = { {	{ 1,        1 / 2.0f, 1 / 4.0f, 1 / 2.0f},
													{ 1 / 2.0f, 1,        1 / 2.0f, 1 / 4.0f},
													{ 1 / 4.0f, 1 / 2.0f, 1 ,       1 / 2.0f},
													{ 1 / 2.0f, 1 / 4.0f, 1 / 2.0f, 1       } } };
	int A = 0;
	int B = 0;
	for (int f = 0; f < 6; f++)
	{
		Eigen::MatrixXf FeConv = matrixBuilder.calculateFaceIntNaNb<ShapeFunctions::HexLinear>(mesh.elements()[0], f);
		// all of these nodes should be scaled relative to scale matrix
		std::array<int,4> nodes = ShapeFunctions::HexLinear::faceConnectivity[f];
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				A = nodes[i];
				B = nodes[j];
				EXPECT_FLOAT_EQ(FeConv(A, B), scale[i][j] / 9.0f) << "A: " << A << " B: " << B;
			}
		}
		// All of these nodes should be on the opposite face so have nodes
		// that should have a 0 value in the matrix
		int f2;
		if (f % 2)
			f2 = (f - 1);
		else
			f2 = (f + 1);
		nodes = ShapeFunctions::HexLinear::faceConnectivity[f2];
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
	matrixBuilder.setNodeMap();
	// only bottom face is dirichlet
	std::vector<long> trueMap = { -1 ,-1,-1,-1, 0, 1, 2, 3 };
	std::vector<long> trueValid = { 4,5,6,7 };
	for (int i = 0; i < 27; i++)
	{
		if (i > 17)
			EXPECT_EQ(-1, matrixBuilder.globalMatrices().nodeMap[i]);
		else
			EXPECT_EQ(i, matrixBuilder.globalMatrices().nodeMap[i]);
	}
	// 27 nodes total, 9 nodes are dirichlet
	EXPECT_EQ(18, matrixBuilder.globalMatrices().nNonDirichlet);
}

TEST_F(HexBuilder, testBuildMatrices)
{
	matrixBuilder.buildMatrices();
	// This test really just makes sure the function runs, we should add checks on the matrix construction
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
	Eigen::Matrix<float,3,Nne> dNdxi;
	for (int a = 0; a < Nne; a++)
	{
		dNdxi.col(a) = ShapeFunctions::TetLinear::dNdxi({1.0f,-1.0f,0.0f}, a);
	}

	Eigen::Matrix3f J = matrixBuilder.calculateJ<ShapeFunctions::TetLinear>(elem, dNdxi); // position (xi) doesn't matter in this test case

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
	matrixBuilder.precomputeElemJ<ShapeFunctions::TetLinear>(mesh.elements()[0]);
	Eigen::Matrix<float, Nne, Nne> Me = matrixBuilder.calculateIntNaNb<ShapeFunctions::TetLinear>(elem);

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
	matrixBuilder.precomputeElemJ<ShapeFunctions::TetLinear>(mesh.elements()[0]);
	Eigen::Matrix<float, Nne, Nne> Ke = matrixBuilder.calculateIntdNadNb<ShapeFunctions::TetLinear>(elem);

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
	Eigen::Vector<float, nNe> FeFlux = matrixBuilder.calculateFaceIntNa<ShapeFunctions::TetLinear>(elem, 0, 1);
	for (int A = 0; A < nNe; A++)
	{
		if (A == 3)
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 2/3.0f);
	}

	//Face 1:
	FeFlux = matrixBuilder.calculateFaceIntNa<ShapeFunctions::TetLinear>(elem, 1, 1);
	for (int A = 0; A < nNe; A++)
	{
		if (A == 2)
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), 2 / 3.0f);
	}

	//Face 2: is slanted and has a larger area
	FeFlux = matrixBuilder.calculateFaceIntNa<ShapeFunctions::TetLinear>(elem, 2, 1);
	for (int A = 0; A < nNe; A++)
	{
		if (A == 0)
			EXPECT_FLOAT_EQ(FeFlux(A), 0);// any input on the top nodes should be 0
		else
			EXPECT_FLOAT_EQ(FeFlux(A), sqrt(3)*2 / 3.0f); // area is larger by factor of sqrt(3) so flux is larger by that amount
	}

	//Face 3:
	FeFlux = matrixBuilder.calculateFaceIntNa<ShapeFunctions::TetLinear>(elem, 3, 1);
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
		Eigen::MatrixXf FeConv = matrixBuilder.calculateFaceIntNaNb<ShapeFunctions::TetLinear>(elem, f);
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

TEST_F(TetBuilder, testBuildMatrices)
{
	matrixBuilder.buildMatrices();
	// This test really just makes sure the function runs, we should add checks on the matrix construction
}