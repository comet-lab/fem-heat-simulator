#include <gtest/gtest.h>
#include "../src/FEM_Simulator.h"
#include "BaseSim.cpp"
#include <iostream>
#include <string>

TEST_F(BaseSim, TestCalculateNABase1)
{
	int Nn1d = 2;
	float xi;
	for (int Ai = 0; Ai < Nn1d; Ai++) {
		xi = -1;
		float output1 = femSimLin->calculateNABase(-1, Ai);
		float output3 = femSimLin->calculateNABase(0, Ai);
		float output2 = femSimLin->calculateNABase(1, Ai);
		if (Ai == 0) {
			EXPECT_FLOAT_EQ(1.0f, output1);
			EXPECT_FLOAT_EQ(0.5f, output3);
			EXPECT_FLOAT_EQ(0.0f, output2);
		}
		else if (Ai == 1) {
			EXPECT_FLOAT_EQ(0.0f, output1);
			EXPECT_FLOAT_EQ(0.5f, output3);
			EXPECT_FLOAT_EQ(1.0f, output2);
		}
	}
}

TEST_F(BaseSim, TestCalculateNABase2)
{
	int Nn1d = 3;
	float xi;
	for (int Ai = 0; Ai < Nn1d; Ai++) {
		xi = -1;

		float output1 = femSimQuad->calculateNABase(-1, Ai);
		float output2 = femSimQuad->calculateNABase(0, Ai);
		float output3 = femSimQuad->calculateNABase(1, Ai);
		float output4 = femSimQuad->calculateNABase(-0.5f, Ai);

		if (Ai == 0) {
			EXPECT_FLOAT_EQ(1.0f, output1);
			EXPECT_FLOAT_EQ(0.375f, output4);
			EXPECT_FLOAT_EQ(0.0f, output2);
			EXPECT_FLOAT_EQ(0.0f, output3);
		}
		else if (Ai == 1) {
			EXPECT_FLOAT_EQ(0.0f, output1);
			EXPECT_FLOAT_EQ(0.75f, output4);
			EXPECT_FLOAT_EQ(1.0f, output2);
			EXPECT_FLOAT_EQ(0.0f, output3);
		}
		else if (Ai == 2) {
			EXPECT_FLOAT_EQ(0.0f, output1);
			EXPECT_FLOAT_EQ(-0.125f, output4);
			EXPECT_FLOAT_EQ(0.0f, output2);
			EXPECT_FLOAT_EQ(1.0f, output3);
		}
	}
}

TEST_F(BaseSim, TestCalculateNA1)
{
	int Nn1d = 2;
	int Nne = pow(Nn1d, 3);
	float xi1[3], xi2[3];
	int AiSub[3];
	int size[3] = { Nn1d,Nn1d,Nn1d };
	for (int Ai = 0; Ai < Nne; Ai++) {
		femSimLin->ind2sub(Ai, size, AiSub);
		xi1[0] = AiSub[0] * 2 - 1;
		xi1[1] = AiSub[1] * 2 - 1;
		xi1[2] = AiSub[2] * 2 - 1;
		xi2[0] = (((AiSub[0] + 1) % 2) * 2 - 1);
		xi2[1] = (((AiSub[1] + 1) % 2) * 2 - 1);;
		xi2[2] = (((AiSub[2] + 1) % 2) * 2 - 1);;
		float output1 = femSimLin->calculateNA(xi1, Ai);
		float output2 = femSimLin->calculateNA(xi2, Ai);

		EXPECT_FLOAT_EQ(1.0f, output1);
		EXPECT_FLOAT_EQ(0.0f, output2);
	}
}

TEST_F(BaseSim, TestCalculateNA2)
{
	int Nn1d = 3;
	int Nne = pow(Nn1d, 3);
	float xi1[3], xi2[3], xi3[3], xi4[3];
	int AiSub[3];
	int size[3] = { Nn1d,Nn1d,Nn1d };
	for (int Ai = 0; Ai < Nne; Ai++) {
		femSimQuad->ind2sub(Ai, size, AiSub);
		xi1[0] = AiSub[0] - 1;
		xi1[1] = AiSub[1] - 1;
		xi1[2] = AiSub[2] - 1;

		xi2[0] = (((AiSub[0] + 1) % Nn1d) - 1);
		xi2[1] = AiSub[1] - 1;
		xi2[2] = AiSub[2] - 1;

		xi3[0] = AiSub[0] - 1;
		xi3[1] = (((AiSub[1] + 1) % Nn1d) - 1);
		xi3[2] = AiSub[2] - 1;

		xi4[0] = AiSub[0] - 1;
		xi4[1] = AiSub[1] - 1;
		xi4[2] = (((AiSub[2] + 1) % Nn1d) - 1);

		float output1 = femSimQuad->calculateNA(xi1, Ai);
		float output2 = femSimQuad->calculateNA(xi2, Ai);
		float output3 = femSimQuad->calculateNA(xi3, Ai);
		float output4 = femSimQuad->calculateNA(xi4, Ai);

		EXPECT_FLOAT_EQ(1.0f, output1);
		EXPECT_FLOAT_EQ(0.0f, output2);
		EXPECT_FLOAT_EQ(0.0f, output3);
		EXPECT_FLOAT_EQ(0.0f, output4);
	}
}

TEST_F(BaseSim, TestcalculateNADotBase1)
{

	int Nn1d = 2;
	for (int Ai = 0; Ai < Nn1d; Ai++) {

		float output1 = femSimLin->calculateNADotBase(-1, Ai);
		float output2 = femSimLin->calculateNADotBase(0, Ai);
		float output3 = femSimLin->calculateNADotBase(1, Ai);
		float output4 = femSimLin->calculateNADotBase(-0.5, Ai);

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

TEST_F(BaseSim, TestcalculateNADotBase2)
{

	int Nn1d = 3;
	for (int Ai = 0; Ai < Nn1d; Ai++) {

		float output1 = femSimQuad->calculateNADotBase(-1, Ai);
		float output2 = femSimQuad->calculateNADotBase(0, Ai);
		float output3 = femSimQuad->calculateNADotBase(1, Ai);
		float output4 = femSimQuad->calculateNADotBase(-0.5, Ai);

		if (Ai == 0) {
			EXPECT_FLOAT_EQ(-3 / 2.0f, output1);
			EXPECT_FLOAT_EQ(-1 / 2.0f, output2);
			EXPECT_FLOAT_EQ(1 / 2.0f, output3);
			EXPECT_FLOAT_EQ(-1.0f, output4);
		}
		else if (Ai == 1) {
			EXPECT_FLOAT_EQ(2.0f, output1);
			EXPECT_FLOAT_EQ(0.0f, output2);
			EXPECT_FLOAT_EQ(-2.0f, output3);
			EXPECT_FLOAT_EQ(1.0f, output4);
		}
		else if (Ai == 2) {
			EXPECT_FLOAT_EQ(-1 / 2.0f, output1);
			EXPECT_FLOAT_EQ(1 / 2.0f, output2);
			EXPECT_FLOAT_EQ(3 / 2.0f, output3);
			EXPECT_FLOAT_EQ(0.0f, output4);
		}
	}
}

TEST_F(BaseSim, TestCalculateNADot2)
{

	int Nn1d = 3;
	int Nne = pow(Nn1d, 3);
	float xi1[3], xi2[3], xi3[3], xi4[3];
	int AiSub[3];
	int size[3] = { Nn1d,Nn1d,Nn1d };
	for (int Ai = 0; Ai < Nne; Ai++) {
		femSimQuad->ind2sub(Ai, size, AiSub);
		xi1[0] = AiSub[0] - 1;
		xi1[1] = AiSub[1] - 1;
		xi1[2] = AiSub[2] - 1;

		xi2[0] = (((AiSub[0] + 1) % Nn1d) - 1);
		xi2[1] = (((AiSub[1] + 1) % Nn1d) - 1);
		xi2[2] = (((AiSub[2] + 1) % Nn1d) - 1);

		Eigen::Vector3f output1 = femSimQuad->calculateNA_dot(xi1, Ai);
		Eigen::Vector3f output2 = femSimQuad->calculateNA_dot(xi2, Ai);

		for (int i = 0; i < 3; i++) {
			if (xi1[i] == -1) {
				EXPECT_FLOAT_EQ(-1.5f, output1(i));
			}
			else if (xi1[i] == 0) {
				EXPECT_FLOAT_EQ(0.0f, output1(i));
			}
			else if (xi1[i] == 1) {
				EXPECT_FLOAT_EQ(1.5f, output1(i));
			}
			EXPECT_FLOAT_EQ(0.0f, output2(i));
		}
	}
}

TEST_F(BaseSim, TestDetermineNodeFace1)
{
	int expected0 = FEM_Simulator::tissueFace::TOP + FEM_Simulator::tissueFace::BACK + FEM_Simulator::tissueFace::LEFT;
	int expected1 = FEM_Simulator::tissueFace::TOP + FEM_Simulator::tissueFace::LEFT;
	int expected2 = FEM_Simulator::tissueFace::TOP + FEM_Simulator::tissueFace::FRONT + FEM_Simulator::tissueFace::LEFT;
	int expected3 = FEM_Simulator::tissueFace::TOP + FEM_Simulator::tissueFace::BACK;
	int expected4 = FEM_Simulator::tissueFace::TOP;
	int expected5 = FEM_Simulator::tissueFace::TOP + FEM_Simulator::tissueFace::FRONT;
	int expected6 = FEM_Simulator::tissueFace::TOP + FEM_Simulator::tissueFace::BACK + FEM_Simulator::tissueFace::RIGHT;
	int expected7 = FEM_Simulator::tissueFace::TOP + FEM_Simulator::tissueFace::RIGHT;
	int expected8 = FEM_Simulator::tissueFace::TOP + FEM_Simulator::tissueFace::FRONT + FEM_Simulator::tissueFace::RIGHT;
	int expected9 = FEM_Simulator::tissueFace::BACK + FEM_Simulator::tissueFace::LEFT;
	int expected10 = FEM_Simulator::tissueFace::LEFT;
	int expected11 = FEM_Simulator::tissueFace::FRONT + FEM_Simulator::tissueFace::LEFT;
	int expected12 = FEM_Simulator::tissueFace::BACK;
	int expected13 = FEM_Simulator::tissueFace::INTERNAL;
	int expected14 = FEM_Simulator::tissueFace::FRONT;
	int expected15 = FEM_Simulator::tissueFace::BACK + FEM_Simulator::tissueFace::RIGHT;
	int expected16 = FEM_Simulator::tissueFace::RIGHT;
	int expected17 = FEM_Simulator::tissueFace::FRONT + FEM_Simulator::tissueFace::RIGHT;
	int expected18 = FEM_Simulator::tissueFace::BOTTOM + FEM_Simulator::tissueFace::BACK + FEM_Simulator::tissueFace::LEFT;
	int expected19 = FEM_Simulator::tissueFace::BOTTOM + FEM_Simulator::tissueFace::LEFT;
	int expected20 = FEM_Simulator::tissueFace::BOTTOM + FEM_Simulator::tissueFace::FRONT + FEM_Simulator::tissueFace::LEFT;
	int expected21 = FEM_Simulator::tissueFace::BOTTOM + FEM_Simulator::tissueFace::BACK;
	int expected22 = FEM_Simulator::tissueFace::BOTTOM;
	int expected23 = FEM_Simulator::tissueFace::BOTTOM + FEM_Simulator::tissueFace::FRONT;
	int expected24 = FEM_Simulator::tissueFace::BOTTOM + FEM_Simulator::tissueFace::BACK + FEM_Simulator::tissueFace::RIGHT;
	int expected25 = FEM_Simulator::tissueFace::BOTTOM + FEM_Simulator::tissueFace::RIGHT;
	int expected26 = FEM_Simulator::tissueFace::BOTTOM + FEM_Simulator::tissueFace::FRONT + FEM_Simulator::tissueFace::RIGHT;
	int expected[27] = { expected0,expected1,expected2,expected3,expected4,expected5,expected6,expected7,expected8,expected9,expected10,
						expected11,expected12,expected13,expected14,expected15,expected16,expected17,expected18,expected19,expected20,
						expected21,expected22,expected23,expected24,expected25,expected26 };

	int faces[27];
	for (int i = 0; i < 27; i++) {
		faces[i] = femSimLin->determineNodeFace(i);
		//std::cout << i << ", Expected: " << expected[i] << " Actual: " << faces[i] << std::endl;
		EXPECT_FLOAT_EQ(expected[i], faces[i]);
	}
}

TEST_F(BaseSim, TestInitializeElementNodeSurfaceMap1)
{

	int Nn1d = 2;
	int Nne = pow(Nn1d, 3);
	femSimLin->initializeElementNodeSurfaceMap();
	std::array<std::vector<int>, 6> surfMap = femSimLin->elemNodeSurfaceMap;


	std::vector<std::vector<bool>> nodeCondition = { { true,false,false,false,true,true }, // Node 0 belongs to the top, back and left faces
													 { true,false,true,false,false,true }, // Node 1 belongs to the top, front, and left faces
													 { true,false,false,true,true,false }, // Node 2 belongs to the top, back, and right faces
													 { true,false,true,true,false,false }, // Node 3 belongs to the top, front, and right faces
													 { false,true,false,false,true,true }, // Node 4 belongs to the bottom, back and left faces
													 { false,true,true,false,false,true }, // Node 5 belongs to the bottom, front, and left faces
													 { false,true,false,true,true,false }, // Node 6 belongs to the bottom, back, and right faces
													 { false,true,true,true,false,false }, // Node 7 belongs to the bottom, front, and right faces
	};
	for (int A = 0; A < Nne; A++)
	{
		for (int f = 0; f < 6; f++) {
			bool contains = (std::find(surfMap[f].begin(), surfMap[f].end(), A) != surfMap[f].end());
			EXPECT_TRUE((contains == nodeCondition[A][f]));
		}
	}
}

TEST_F(BaseSim, TestInitializeBoundaryNodes1)
{

	int Nn1d = 2;
	int Nne = pow(Nn1d, 3);
	int BC[6] = { 0,0,0,0,0,0 }; // All nodes except the one in the center should be dirichlet nodes 
	femSimLin->setBoundaryConditions(BC);

	femSimLin->initializeBoundaryNodes();

	for (int idx = 0; idx < 27; idx++) {
		std::vector<int> dirchletNodes = femSimLin->dirichletNodes;
		std::vector<int> validNodes = femSimLin->validNodes;
		std::vector<int> nodeMap = femSimLin->nodeMap;
		bool dirContains = (std::find(dirchletNodes.begin(), dirchletNodes.end(), idx) != dirchletNodes.end());
		bool validContains = (std::find(validNodes.begin(), validNodes.end(), idx) != validNodes.end());
		if (idx != 13) {
			EXPECT_TRUE(dirContains);
			EXPECT_FALSE(validContains);
			EXPECT_FLOAT_EQ(-1, nodeMap[idx]);
		}
		else {
			// Node 13 is in the middle of everything and is the exception
			EXPECT_FALSE(dirContains);
			EXPECT_TRUE(validContains);
			EXPECT_FLOAT_EQ(0, nodeMap[idx]);
		}
	}
}

TEST_F(BaseSim, TestInitializeBoundaryNodes2)
{

	int Nn1d = 2;
	int Nne = pow(Nn1d, 3);
	int BC[6] = { 0,0,0,1,2,1 }; // All Nodes on top/bottom/front faces are dirichlet
	femSimLin->setBoundaryConditions(BC);

	femSimLin->initializeBoundaryNodes();

	bool dirichletCond[27] = { true,true,true,true,true,true,true,true,true, // top layer
							  false,false,true,false,false,true,false,false,true, //middle layer
							  true,true,true,true,true,true,true,true,true }; // bottom layer
	int trueMap[27] = { -1,-1,-1,-1,-1,-1,-1,-1,-1, // top layer
							  0,1,-1,2,3,-1,4,5,-1, //middle layer
							  -1,-1,-1,-1,-1,-1,-1,-1,-1 }; // bottom layer
	for (int idx = 0; idx < 27; idx++) {
		std::vector<int> dirchletNodes = femSimLin->dirichletNodes;
		std::vector<int> validNodes = femSimLin->validNodes;
		std::vector<int> nodeMap = femSimLin->nodeMap;
		bool dirContains = (std::find(dirchletNodes.begin(), dirchletNodes.end(), idx) != dirchletNodes.end());
		bool validContains = (std::find(validNodes.begin(), validNodes.end(), idx) != validNodes.end());

		EXPECT_TRUE(dirContains == dirichletCond[idx]);
		EXPECT_TRUE(validContains == (!dirichletCond[idx]));
		EXPECT_FLOAT_EQ(trueMap[idx], nodeMap[idx]);
	}
}

TEST_F(BaseSim, TestInitializeBoundaryNodes3)
{

	int BC[6] = { 1,2,2,0,2,1 }; // All Nodes on top/bottom/front faces are dirichlet
	femSimLin->setBoundaryConditions(BC);

	femSimLin->initializeBoundaryNodes();

	bool dirichletCond[27] = { false,false,false,false,false,false,true,true,true, // top layer
							  false,false,false,false,false,false,true,true,true, //middle layer
							  false,false,false,false,false,false,true,true,true }; // bottom layer
	int trueMap[27] = { 0,1,2,3,4,5,-1,-1,-1, // top layer
							  6,7,8,9,10,11,-1,-1,-1, //middle layer
							  12,13,14,15,16,17,-1,-1,-1 }; // bottom layer
	for (int idx = 0; idx < 27; idx++) {
		std::vector<int> dirchletNodes = femSimLin->dirichletNodes;
		std::vector<int> validNodes = femSimLin->validNodes;
		std::vector<int> nodeMap = femSimLin->nodeMap;
		bool dirContains = (std::find(dirchletNodes.begin(), dirchletNodes.end(), idx) != dirchletNodes.end());
		bool validContains = (std::find(validNodes.begin(), validNodes.end(), idx) != validNodes.end());

		EXPECT_TRUE(dirContains == dirichletCond[idx]);
		EXPECT_TRUE(validContains == (!dirichletCond[idx]));
		EXPECT_FLOAT_EQ(trueMap[idx], nodeMap[idx]);
	}
}

TEST_F(BaseSim, TestInitializeBoundaryNodes4)
{
	int BC[6] = { 1,2,2,1,2,1 }; // All Nodes on top/bottom/front faces are dirichlet
	femSimLin->setBoundaryConditions(BC);

	femSimLin->initializeBoundaryNodes();

	for (int idx = 0; idx < 27; idx++) {
		std::vector<int> dirchletNodes = femSimLin->dirichletNodes;
		std::vector<int> validNodes = femSimLin->validNodes;
		std::vector<int> nodeMap = femSimLin->nodeMap;
		bool dirContains = (std::find(dirchletNodes.begin(), dirchletNodes.end(), idx) != dirchletNodes.end());
		bool validContains = (std::find(validNodes.begin(), validNodes.end(), idx) != validNodes.end());

		EXPECT_TRUE(dirContains == false);
		EXPECT_TRUE(validContains == true);
		EXPECT_FLOAT_EQ(idx, nodeMap[idx]);
	}
}

TEST_F(BaseSim, TestCalculateJ1)
{
	
	float tissueSize[3] = { 2,1,0.5 };

	femSimLin->setTissueSize(tissueSize);
	femSimLin->setJ(1);

	EXPECT_FLOAT_EQ(tissueSize[0] / 4.0f, femSimLin->J(0, 0));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->J(0, 1));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->J(0, 2));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->J(1, 0));
	EXPECT_FLOAT_EQ(tissueSize[1] / 4.0f, femSimLin->J(1, 1));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->J(1, 2));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->J(2, 0));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->J(2, 1));
	EXPECT_FLOAT_EQ(tissueSize[2] / 4.0f, femSimLin->J(2, 2));
}

TEST_F(BaseSim, TestCalculateJs1_1)
{

	float tissueSize[3] = { 2,1,0.5 };
	femSimLin->setTissueSize(tissueSize);
	femSimLin->setJ();

	EXPECT_FLOAT_EQ(tissueSize[1] / 4, femSimLin->Js1(0, 0));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->Js1(0, 1));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->Js1(1, 0));
	EXPECT_FLOAT_EQ(tissueSize[2] / 4, femSimLin->Js1(1, 1));
}

TEST_F(BaseSim, TestCalculateJs2_1)
{
	float tissueSize[3] = { 2,1,0.5 };
	femSimLin->setTissueSize(tissueSize);
	femSimLin->setJ();

	EXPECT_FLOAT_EQ(tissueSize[0] / 4, femSimLin->Js2(0, 0));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->Js2(0, 1));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->Js2(1, 0));
	EXPECT_FLOAT_EQ(tissueSize[2] / 4, femSimLin->Js2(1, 1));
}

TEST_F(BaseSim, TestCalculateJs3_1)
{
	float tissueSize[3] = { 2,1,0.5 };
	femSimLin->setTissueSize(tissueSize);
	femSimLin->setJ();

	EXPECT_FLOAT_EQ(tissueSize[0] / 4, femSimLin->Js3(0, 0));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->Js3(0, 1));
	EXPECT_FLOAT_EQ(0.0f, femSimLin->Js3(1, 0));
	EXPECT_FLOAT_EQ(tissueSize[1] / 4, femSimLin->Js3(1, 1));
}

TEST_F(BaseSim, TestInd2Sub1)
{
	

	int sub[3];
	int index = 0;
	int size[3] = { 10,10,10 };
	femSimLin->ind2sub(index, size, sub);
	EXPECT_FLOAT_EQ(0, sub[0]);
	EXPECT_FLOAT_EQ(0, sub[1]);
	EXPECT_FLOAT_EQ(0, sub[2]);
}

TEST_F(BaseSim, TestInd2Sub2)
{

	int sub[3];
	int index = 10;
	int size[3] = { 10,10,10 };
	femSimLin->ind2sub(index, size, sub);
	EXPECT_FLOAT_EQ(0, sub[0]);
	EXPECT_FLOAT_EQ(1, sub[1]);
	EXPECT_FLOAT_EQ(0, sub[2]);
}

TEST_F(BaseSim, TestInd2Sub3)
{

	int sub[3];
	int index = 100;
	int size[3] = { 10,10,10 };
	femSimLin->ind2sub(index, size, sub);
	EXPECT_FLOAT_EQ(0, sub[0]);
	EXPECT_FLOAT_EQ(0, sub[1]);
	EXPECT_FLOAT_EQ(1, sub[2]);
}

TEST_F(BaseSim, TestInd2Sub4)
{
	
	int sub[3];
	int index = 521;
	int size[3] = { 10,10,10 };
	femSimLin->ind2sub(index, size, sub);
	EXPECT_FLOAT_EQ(1, sub[0]);
	EXPECT_FLOAT_EQ(2, sub[1]);
	EXPECT_FLOAT_EQ(5, sub[2]);
}


TEST_F(BaseSim, TestCalcKABFunction1)
{

	float TC = femSimLin->TC;
	int Ai = 0;
	int Bi = 0;
	float xi[3];
	xi[0] = FEM_Simulator::A[Ai][0];
	xi[1] = FEM_Simulator::A[Ai][1];
	xi[2] = FEM_Simulator::A[Ai][2];
	float output1 = femSimLin->calcKintAB(xi, Ai, Bi);
	Eigen::MatrixXf KeInt(8, 8);
	for (int Ai = 0; Ai < 8; Ai++) {
		for (int Bi = 0; Bi < 8; Bi++) {
			KeInt(Ai, Bi) = femSimLin->integrate(&FEM_Simulator::calcKintAB, 2, 0, Ai, Bi);
		}
	}
	// The truth values were calculated in matlab assuming Kint = 1 and deltaX = deltaY = deltaZ = 0.5
	EXPECT_TRUE((abs(3 / 16.0f) - output1) < 0.0001);
	for (int Ai = 0; Ai < 8; Ai++) {
		EXPECT_TRUE(((1 / 6.0f ) - KeInt(Ai, Ai)) < 0.0001);
	}
	EXPECT_TRUE(((-1 / 24.0f) - KeInt(2, 0)) < 0.0001);
}

TEST_F(BaseSim, TestCreateKABFunction2)
{
	int Ai = 0;
	int Bi = 0;
	float xi[3];
	xi[0] = -1;
	xi[1] = -1;
	xi[2] = -1;
	float output1 = femSimQuad->calcKintAB(xi, Ai, Bi);
	femSimQuad->setKeInt();
	
	EXPECT_TRUE(abs(3.375f - output1) < 0.0001);
	EXPECT_TRUE(abs(0.1244f - femSimQuad->KeInt(0, 0)) < 0.0001);
	EXPECT_TRUE(abs(0.4267f - femSimQuad->KeInt(1, 1)) < 0.0001);
	EXPECT_TRUE(abs(1.4222f - femSimQuad->KeInt(4, 4)) < 0.0001);
	EXPECT_TRUE(abs(4.5511f - femSimQuad->KeInt(13, 13)) < 0.0001);
	EXPECT_TRUE(abs(-0.0415f - femSimQuad->KeInt(9, 15)) < 0.0001);
	EXPECT_TRUE(abs(-0.0059f - femSimQuad->KeInt(21, 5)) < 0.0001);
	EXPECT_TRUE(abs(-0.0370f - femSimQuad->KeInt(21, 15)) < 0.0001);
}

TEST_F(BaseSim, TestCreateMABFunction1)
{
	
	int Ai = 0;
	int Bi = 0;
	float xi[3];
	xi[0] = -1;
	xi[1] = -1;
	xi[2] = -1;
	float output1 = femSimLin->calcMAB(xi, Ai, Bi);
	Eigen::MatrixXf Me(8, 8);
	for (int Ai = 0; Ai < 8; Ai++) {
		for (int Bi = 0; Bi < 8; Bi++) {
			Me(Ai, Bi) = femSimLin->integrate(&FEM_Simulator::calcMAB, 2, 0, Ai, Bi);
		}
	}
	// The truth values were calculated in matlab assuming Kint = 1 and deltaX = deltaY = deltaZ = 0.5
	EXPECT_TRUE((abs(1 / 64.0f) - output1) < 0.0001);
	for (int Ai = 0; Ai < 8; Ai++) {
		EXPECT_TRUE(((1 / 216.0f) - Me(Ai, Ai)) < 0.0001);
	}
	EXPECT_TRUE(((1 / 864.0f) - Me(2, 0)) < 0.0001);
}

TEST_F(BaseSim, TestCreateFjFunction1)
{
	int Ai = 0;
	int Bi = 0;
	float xi[3];
	xi[0] = FEM_Simulator::A[Ai][0];
	xi[1] = FEM_Simulator::A[Ai][1];
	xi[2] = FEM_Simulator::A[Ai][2];
	float output1 = femSimLin->calcFqA(xi, Ai, 1);
	femSimLin->setFeQ();
	// The truth values were calculated in matlab assuming Kint = 1 and deltaX = deltaY = deltaZ = 0.5
	//EXPECT_TRUE((abs(1 / 64.0f) - output1) < 0.0001);
	//for (int Ai = 0; Ai < 8; Ai++) {
	//	EXPECT_TRUE(((1 / 216.0f) - Me(Ai, Ai)) < 0.0001);
	//}
	//EXPECT_TRUE(((1 / 864.0f * TC) - Me(2, 0)) < 0.0001);
}

TEST_F(BaseSim, CompareLinearAndQuadratic1) {

	int nodesPerAxis[3] = { 5,5,5 };
	std::vector<std::vector<std::vector<float>>> Temp(nodesPerAxis[0], std::vector<std::vector<float>>(nodesPerAxis[1], std::vector<float>(nodesPerAxis[2])));
	std::vector<std::vector<std::vector<float>>> NFR(nodesPerAxis[0], std::vector<std::vector<float>>(nodesPerAxis[1], std::vector<float>(nodesPerAxis[2])));
	srand(1);
	for (int i = 0; i < nodesPerAxis[0]; i++) {
		for (int j = 0; j < nodesPerAxis[1]; j++) {
			for (int k = 0; k < nodesPerAxis[2]; k++) {
				Temp[i][j][k] = 0;
				NFR[i][j][k] = 1;
			}
		}
	}
	float tissueSize[3] = { 1,1,1 };
	float TC = 1.0f;
	int BC[6] = { 0,0,2,2,2,2 };
	FEM_Simulator* femSimLin = new FEM_Simulator(Temp, tissueSize, TC, 1.0f, 1.0f, 1.0f, 2);
	FEM_Simulator* femSimQuad = new FEM_Simulator(Temp, tissueSize, TC, 1.0f, 1.0f, 1.0f, 3);

	femSimLin->deltaT = 0.05f;
	femSimLin->tFinal = 1.0f;
	femSimLin->setBoundaryConditions(BC);
	femSimLin->setFlux(0);
	femSimLin->setAmbientTemp(0);
	femSimLin->setNFR(NFR);

	femSimQuad->deltaT = 0.05f;
	femSimQuad->tFinal = 1.0f;
	femSimQuad->setBoundaryConditions(BC);
	femSimQuad->setFlux(0);
	femSimQuad->setAmbientTemp(0);
	femSimQuad->setNFR(NFR);

	femSimLin->createKMFelem();
	femSimLin->performTimeStepping();
	femSimQuad->createKMFelem();
	femSimQuad->performTimeStepping();

	for (int k = 0; k < nodesPerAxis[2]; k++) {
		for (int j = 0; j < nodesPerAxis[1]; j++) {
			for (int i = 0; i < nodesPerAxis[0]; i++) {
				int idx = i + j * nodesPerAxis[0] + k * nodesPerAxis[0] * nodesPerAxis[1];
				EXPECT_TRUE(abs(femSimQuad->Temp(idx) - femSimQuad->Temp(idx)) < 0.001);
			}
		}
	}

}

TEST_F(BaseSim, testPositionToElement1) {
	// only single layer test
	std::array<std::array<float, 3>, 5> testPositions = { {
		{{ -0.5, -0.5, 0 }},
		{{ 0, 0, 0 }},
		{{ -0.25, -0.25, 0.75 }},
		{{ 0.5, 0.5, 1 }},
		{{ 0.1, -0.2, 0.2 }}
		} };
	std::array<std::array<int, 3>, 5> exOutElement = { {
		{{ 0, 0, 0}},
		{{ 1, 1, 0}},
		{{ 0, 0, 1}},
		{{ 1, 1, 1}},
		{{ 1, 0, 0}}
		} };
	std::array<std::array<float, 3>, 5> exOutXi = { {
		{{ -1, -1, -1}},
		{{ -1, -1, -1}},
		{{ 0, 0, 0}},
		{{ 1, 1, 1}},
		{{-0.6, 0.2,-0.2}}
		} };

	for (int i = 0; i < 5; i++) {
		float xiOutput[3];
		std::array<int, 3> elementOutput;
		elementOutput = femSimLin->positionToElement(testPositions[i], xiOutput);
		for (int j = 0; j < 3; j++) {
			ASSERT_EQ(exOutElement[i][j], elementOutput[j]);
			ASSERT_FLOAT_EQ(exOutXi[i][j], xiOutput[j]); //abs(exOutXi[i][j] - xiOutput[j]) < 0.00001
		}
	}
}

TEST(SecondaryTest, testPositionToElement2) {

	// Multi Layer Test

	int nodesPerAxis[3] = {21,21,20};
	std::vector<std::vector<std::vector<float>>> Temp(nodesPerAxis[0], std::vector<std::vector<float>>(nodesPerAxis[1], std::vector<float>(nodesPerAxis[2])));
	for (int i = 0; i < nodesPerAxis[0]; i++) {
		for (int j = 0; j < nodesPerAxis[1]; j++) {
			for (int k = 0; k < nodesPerAxis[2]; k++)
				Temp[i][j][k] = 0;
		}
	}
	float layerHeight = 0.1;
	int layerSize = 10;
	float tissueSize[3] = { 2,2,1 };
	int Nn1d = 2;
	float mua = 1.0f;
	float tc = 1.0f;
	float vhc = 1.0f;
	float htc = 1.0f;

	FEM_Simulator* femSimLin = new FEM_Simulator(Temp, tissueSize, tc, vhc, mua, htc, Nn1d);
	femSimLin->setLayer(layerHeight, layerSize);


	std::array<std::array<float, 3>,7> testPositions = { {
		{{ -1, -1, 0 }},
		{{ 0, 0, 0 }},
		{{ -0.25, -0.25, 0.02 }},
		{{ 1, 1, 1 }},
		{{ -0.25, -0.25, 0.15 }},
		{{ -0.25, 0.25, 0.5 }},
		{{ 0.25, -0.25, 0.045 }}
		} };
	std::array<std::array<int, 3>, 7> exOutElement = { {
		{{ 0, 0, 0}},
		{{ 10, 10, 0}},
		{{ 7, 7, 2}},
		{{ 19,19,18}},
		{{ 7, 7, 10}},
		{{ 7, 12, 14}},
		{{ 12, 7, 4}}
		} };
	std::array<std::array<float, 3>, 7> exOutXi = { {
		{{ -1, -1, -1}},
		{{ -1, -1, -1}},
		{{ 0, 0, -1}},
		{{ 1, 1, 1}},
		{{ 0, 0, 0}},
		{{ 0, 0, -1}},
		{{ 0, 0, 0}}
		} };

	for (int i = 0; i < 7; i++) {
		float xiOutput[3];
		std::array<int, 3> elementOutput;
		elementOutput = femSimLin->positionToElement(testPositions[i], xiOutput);
		for (int j = 0; j < 3; j++) {
			ASSERT_EQ(exOutElement[i][j], elementOutput[j]);
			ASSERT_TRUE(abs(exOutXi[i][j] - xiOutput[j]) < 0.00001); //
		}
	}

}