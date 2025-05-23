#include <gtest/gtest.h>
#include "../src/FEM_Simulator.h"
#include <iostream>
#include <string>

TEST(MainTest, TestCalculateNABase1)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 2;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f, Nn1d);
	float xi;
	for (int Ai = 0; Ai < Nn1d; Ai++) {
		xi = -1;
		float output1 = simulator->calculateNABase(-1, Ai);
		float output3 = simulator->calculateNABase(0, Ai);
		float output2 = simulator->calculateNABase(1, Ai);
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

TEST(MainTest, TestCalculateNABase2)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 3;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f, Nn1d);
	float xi;
	for (int Ai = 0; Ai < Nn1d; Ai++) {
		xi = -1;

		float output1 = simulator->calculateNABase(-1, Ai);
		float output2 = simulator->calculateNABase(0, Ai);
		float output3 = simulator->calculateNABase(1, Ai);
		float output4 = simulator->calculateNABase(-0.5f, Ai);

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

TEST(MainTest, TestCalculateNA1)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 2;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f, Nn1d);

	int Nne = pow(Nn1d, 3);
	float xi1[3], xi2[3];
	int AiSub[3];
	int size[3] = { Nn1d,Nn1d,Nn1d };
	for (int Ai = 0; Ai < Nne; Ai++) {
		simulator->ind2sub(Ai, size, AiSub);
		xi1[0] = AiSub[0] * 2 - 1;
		xi1[1] = AiSub[1] * 2 - 1;
		xi1[2] = AiSub[2] * 2 - 1;
		xi2[0] = (((AiSub[0] + 1) % 2) * 2 - 1);
		xi2[1] = (((AiSub[1] + 1) % 2) * 2 - 1);;
		xi2[2] = (((AiSub[2] + 1) % 2) * 2 - 1);;
		float output1 = simulator->calculateNA(xi1, Ai);
		float output2 = simulator->calculateNA(xi2, Ai);

		EXPECT_FLOAT_EQ(1.0f, output1);
		EXPECT_FLOAT_EQ(0.0f, output2);
	}
}

TEST(MainTest, TestCalculateNA2)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 3;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f, Nn1d);

	int Nne = pow(Nn1d, 3);
	float xi1[3], xi2[3], xi3[3], xi4[3];
	int AiSub[3];
	int size[3] = { Nn1d,Nn1d,Nn1d };
	for (int Ai = 0; Ai < Nne; Ai++) {
		simulator->ind2sub(Ai, size, AiSub);
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

		float output1 = simulator->calculateNA(xi1, Ai);
		float output2 = simulator->calculateNA(xi2, Ai);
		float output3 = simulator->calculateNA(xi3, Ai);
		float output4 = simulator->calculateNA(xi4, Ai);

		EXPECT_FLOAT_EQ(1.0f, output1);
		EXPECT_FLOAT_EQ(0.0f, output2);
		EXPECT_FLOAT_EQ(0.0f, output3);
		EXPECT_FLOAT_EQ(0.0f, output4);
	}
}

TEST(MainTest, TestcalculateNADotBase1)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 2;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f, Nn1d);

	for (int Ai = 0; Ai < Nn1d; Ai++) {

		float output1 = simulator->calculateNADotBase(-1, Ai);
		float output2 = simulator->calculateNADotBase(0, Ai);
		float output3 = simulator->calculateNADotBase(1, Ai);
		float output4 = simulator->calculateNADotBase(-0.5, Ai);

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

TEST(MainTest, TestcalculateNADotBase2)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 3;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f, Nn1d);

	for (int Ai = 0; Ai < Nn1d; Ai++) {

		float output1 = simulator->calculateNADotBase(-1, Ai);
		float output2 = simulator->calculateNADotBase(0, Ai);
		float output3 = simulator->calculateNADotBase(1, Ai);
		float output4 = simulator->calculateNADotBase(-0.5, Ai);

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

TEST(MainTest, TestCalculateNADot2)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 3;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f, Nn1d);

	int Nne = pow(Nn1d, 3);
	float xi1[3], xi2[3], xi3[3], xi4[3];
	int AiSub[3];
	int size[3] = { Nn1d,Nn1d,Nn1d };
	for (int Ai = 0; Ai < Nne; Ai++) {
		simulator->ind2sub(Ai, size, AiSub);
		xi1[0] = AiSub[0] - 1;
		xi1[1] = AiSub[1] - 1;
		xi1[2] = AiSub[2] - 1;

		xi2[0] = (((AiSub[0] + 1) % Nn1d) - 1);
		xi2[1] = (((AiSub[1] + 1) % Nn1d) - 1);
		xi2[2] = (((AiSub[2] + 1) % Nn1d) - 1);

		Eigen::Vector3f output1 = simulator->calculateNA_dot(xi1, Ai);
		Eigen::Vector3f output2 = simulator->calculateNA_dot(xi2, Ai);

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

TEST(MainTest, TestDetermineNodeFace1)
{

	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);

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
		faces[i] = simulator->determineNodeFace(i);
		//std::cout << i << ", Expected: " << expected[i] << " Actual: " << faces[i] << std::endl;
		EXPECT_FLOAT_EQ(expected[i], faces[i]);
	}
}

TEST(MainTest, TestInitializeElementNodeSurfaceMap1)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 2;
	int Nne = pow(Nn1d, 3);
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
	simulator->Nn1d = 2;
	simulator->initializeElementNodeSurfaceMap();
	std::array<std::vector<int>, 6> surfMap = simulator->elemNodeSurfaceMap;


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

TEST(MainTest, TestInitializeBoundaryNodes1)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 2;
	int Nne = pow(Nn1d, 3);
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
	int BC[6] = { 0,0,0,0,0,0 }; // All nodes except the one in the center should be dirichlet nodes 
	simulator->setBoundaryConditions(BC);

	simulator->initializeBoundaryNodes();

	for (int idx = 0; idx < 27; idx++) {
		std::vector<int> dirchletNodes = simulator->dirichletNodes;
		std::vector<int> validNodes = simulator->validNodes;
		std::vector<int> nodeMap = simulator->nodeMap;
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

TEST(MainTest, TestInitializeBoundaryNodes2)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 2;
	int Nne = pow(Nn1d, 3);
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
	int BC[6] = { 0,0,0,1,2,1 }; // All Nodes on top/bottom/front faces are dirichlet
	simulator->setBoundaryConditions(BC);

	simulator->initializeBoundaryNodes();

	bool dirichletCond[27] = { true,true,true,true,true,true,true,true,true, // top layer
							  false,false,true,false,false,true,false,false,true, //middle layer
							  true,true,true,true,true,true,true,true,true }; // bottom layer
	int trueMap[27] = { -1,-1,-1,-1,-1,-1,-1,-1,-1, // top layer
							  0,1,-1,2,3,-1,4,5,-1, //middle layer
							  -1,-1,-1,-1,-1,-1,-1,-1,-1 }; // bottom layer
	for (int idx = 0; idx < 27; idx++) {
		std::vector<int> dirchletNodes = simulator->dirichletNodes;
		std::vector<int> validNodes = simulator->validNodes;
		std::vector<int> nodeMap = simulator->nodeMap;
		bool dirContains = (std::find(dirchletNodes.begin(), dirchletNodes.end(), idx) != dirchletNodes.end());
		bool validContains = (std::find(validNodes.begin(), validNodes.end(), idx) != validNodes.end());

		EXPECT_TRUE(dirContains == dirichletCond[idx]);
		EXPECT_TRUE(validContains == (!dirichletCond[idx]));
		EXPECT_FLOAT_EQ(trueMap[idx], nodeMap[idx]);
	}
}

TEST(MainTest, TestInitializeBoundaryNodes3)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 2;
	int Nne = pow(Nn1d, 3);
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
	int BC[6] = { 1,2,2,0,2,1 }; // All Nodes on top/bottom/front faces are dirichlet
	simulator->setBoundaryConditions(BC);

	simulator->initializeBoundaryNodes();

	bool dirichletCond[27] = { false,false,false,false,false,false,true,true,true, // top layer
							  false,false,false,false,false,false,true,true,true, //middle layer
							  false,false,false,false,false,false,true,true,true }; // bottom layer
	int trueMap[27] = { 0,1,2,3,4,5,-1,-1,-1, // top layer
							  6,7,8,9,10,11,-1,-1,-1, //middle layer
							  12,13,14,15,16,17,-1,-1,-1 }; // bottom layer
	for (int idx = 0; idx < 27; idx++) {
		std::vector<int> dirchletNodes = simulator->dirichletNodes;
		std::vector<int> validNodes = simulator->validNodes;
		std::vector<int> nodeMap = simulator->nodeMap;
		bool dirContains = (std::find(dirchletNodes.begin(), dirchletNodes.end(), idx) != dirchletNodes.end());
		bool validContains = (std::find(validNodes.begin(), validNodes.end(), idx) != validNodes.end());

		EXPECT_TRUE(dirContains == dirichletCond[idx]);
		EXPECT_TRUE(validContains == (!dirichletCond[idx]));
		EXPECT_FLOAT_EQ(trueMap[idx], nodeMap[idx]);
	}
}

TEST(MainTest, TestInitializeBoundaryNodes4)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	int Nn1d = 2;
	int Nne = pow(Nn1d, 3);
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
	int BC[6] = { 1,2,2,1,2,1 }; // All Nodes on top/bottom/front faces are dirichlet
	simulator->setBoundaryConditions(BC);

	simulator->initializeBoundaryNodes();

	for (int idx = 0; idx < 27; idx++) {
		std::vector<int> dirchletNodes = simulator->dirichletNodes;
		std::vector<int> validNodes = simulator->validNodes;
		std::vector<int> nodeMap = simulator->nodeMap;
		bool dirContains = (std::find(dirchletNodes.begin(), dirchletNodes.end(), idx) != dirchletNodes.end());
		bool validContains = (std::find(validNodes.begin(), validNodes.end(), idx) != validNodes.end());

		EXPECT_TRUE(dirContains == false);
		EXPECT_TRUE(validContains == true);
		EXPECT_FLOAT_EQ(idx, nodeMap[idx]);
	}
}

TEST(MainTest, TestCalculateJ1)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 2,1,0.5 };
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);

	EXPECT_FLOAT_EQ(tissueSize[0] / 4, simulator->J(0, 0));
	EXPECT_FLOAT_EQ(0.0f, simulator->J(0, 1));
	EXPECT_FLOAT_EQ(0.0f, simulator->J(0, 2));
	EXPECT_FLOAT_EQ(0.0f, simulator->J(1, 0));
	EXPECT_FLOAT_EQ(tissueSize[1] / 4, simulator->J(1, 1));
	EXPECT_FLOAT_EQ(0.0f, simulator->J(1, 2));
	EXPECT_FLOAT_EQ(0.0f, simulator->J(2, 0));
	EXPECT_FLOAT_EQ(0.0f, simulator->J(2, 1));
	EXPECT_FLOAT_EQ(tissueSize[2] / 4, simulator->J(2, 2));
}

TEST(MainTest, TestCalculateJs1_1)
{
	// Js1 is the partial jacobian for the front/back faces. aka the y-z plane
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 2,1,0.5 };
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);

	EXPECT_FLOAT_EQ(tissueSize[1] / 4, simulator->Js1(0, 0));
	EXPECT_FLOAT_EQ(0.0f, simulator->Js1(0, 1));
	EXPECT_FLOAT_EQ(0.0f, simulator->Js1(1, 0));
	EXPECT_FLOAT_EQ(tissueSize[2] / 4, simulator->Js1(1, 1));
}

TEST(MainTest, TestCalculateJs2_1)
{
	// Js2 is the partial jacobian for the left/right faces. aka the x-z plane
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 2,1,0.5 };
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);

	EXPECT_FLOAT_EQ(tissueSize[0] / 4, simulator->Js2(0, 0));
	EXPECT_FLOAT_EQ(0.0f, simulator->Js2(0, 1));
	EXPECT_FLOAT_EQ(0.0f, simulator->Js2(1, 0));
	EXPECT_FLOAT_EQ(tissueSize[2] / 4, simulator->Js2(1, 1));
}

TEST(MainTest, TestCalculateJs3_1)
{
	// Js3 is the partial jacobian for the top/bottom faces. aka the x-y plane
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 2,1,0.5 };
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);

	EXPECT_FLOAT_EQ(tissueSize[0] / 4, simulator->Js3(0, 0));
	EXPECT_FLOAT_EQ(0.0f, simulator->Js3(0, 1));
	EXPECT_FLOAT_EQ(0.0f, simulator->Js3(1, 0));
	EXPECT_FLOAT_EQ(tissueSize[1] / 4, simulator->Js3(1, 1));
}

TEST(MainTest, TestInd2Sub1)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 2,1,0.5 };
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
	int sub[3];
	int index = 0;
	int size[3] = { 10,10,10 };
	simulator->ind2sub(index, size, sub);
	EXPECT_FLOAT_EQ(0, sub[0]);
	EXPECT_FLOAT_EQ(0, sub[1]);
	EXPECT_FLOAT_EQ(0, sub[2]);
}

TEST(MainTest, TestInd2Sub2)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 2,1,0.5 };
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
	int sub[3];
	int index = 10;
	int size[3] = { 10,10,10 };
	simulator->ind2sub(index, size, sub);
	EXPECT_FLOAT_EQ(0, sub[0]);
	EXPECT_FLOAT_EQ(1, sub[1]);
	EXPECT_FLOAT_EQ(0, sub[2]);
}

TEST(MainTest, TestInd2Sub3)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 2,1,0.5 };
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
	int sub[3];
	int index = 100;
	int size[3] = { 10,10,10 };
	simulator->ind2sub(index, size, sub);
	EXPECT_FLOAT_EQ(0, sub[0]);
	EXPECT_FLOAT_EQ(0, sub[1]);
	EXPECT_FLOAT_EQ(1, sub[2]);
}

TEST(MainTest, TestInd2Sub4)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 2,1,0.5 };
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
	int sub[3];
	int index = 521;
	int size[3] = { 10,10,10 };
	simulator->ind2sub(index, size, sub);
	EXPECT_FLOAT_EQ(1, sub[0]);
	EXPECT_FLOAT_EQ(2, sub[1]);
	EXPECT_FLOAT_EQ(5, sub[2]);
}


TEST(MainTest, TestCalcKABFunction1)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	float TC = 1.0f;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, TC, 1.0f, 1.0f, 1.0f);
	int Ai = 0;
	int Bi = 0;
	float xi[3];
	xi[0] = FEM_Simulator::A[Ai][0];
	xi[1] = FEM_Simulator::A[Ai][1];
	xi[2] = FEM_Simulator::A[Ai][2];
	float output1 = simulator->calcKintAB(xi, Ai, Bi);
	Eigen::MatrixXf KeInt(8, 8);
	for (int Ai = 0; Ai < 8; Ai++) {
		for (int Bi = 0; Bi < 8; Bi++) {
			KeInt(Ai, Bi) = simulator->integrate(&FEM_Simulator::calcKintAB, 2, 0, Ai, Bi);
		}
	}
	// The truth values were calculated in matlab assuming Kint = 1 and deltaX = deltaY = deltaZ = 0.5
	EXPECT_TRUE((abs(3 * TC / 16.0f) - output1) < 0.0001);
	for (int Ai = 0; Ai < 8; Ai++) {
		EXPECT_TRUE(((1 / 6.0f * TC) - KeInt(Ai, Ai)) < 0.0001);
	}
	EXPECT_TRUE(((-1 / 24.0f * TC) - KeInt(2, 0)) < 0.0001);
}

TEST(MainTest, TestCreateKABFunction2)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	float TC = 1.0f;
	int Nn1d = 3;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, TC, 1.0f, 1.0f, 1.0f, Nn1d);
	int Ai = 0;
	int Bi = 0;
	float xi[3];
	xi[0] = -1;
	xi[1] = -1;
	xi[2] = -1;
	float output1 = simulator->calcKintAB(xi, Ai, Bi);
	simulator->setKeInt();
	// The truth values were calculated in matlab assuming Kint = 1 and deltaX = deltaY = deltaZ = 1.0
	EXPECT_TRUE(abs(3.375f - output1) < 0.0001);
	EXPECT_TRUE(abs(0.1244f - simulator->KeInt(0, 0)) < 0.0001);
	EXPECT_TRUE(abs(0.4267f - simulator->KeInt(1, 1)) < 0.0001);
	EXPECT_TRUE(abs(1.4222f - simulator->KeInt(4, 4)) < 0.0001);
	EXPECT_TRUE(abs(4.5511f - simulator->KeInt(13, 13)) < 0.0001);
	EXPECT_TRUE(abs(-0.0415f - simulator->KeInt(9, 15)) < 0.0001);
	EXPECT_TRUE(abs(-0.0059f - simulator->KeInt(21, 5)) < 0.0001);
	EXPECT_TRUE(abs(-0.0370f - simulator->KeInt(21, 15)) < 0.0001);
}

TEST(MainTest, TestCreateMABFunction1)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	float TC = 1.0f;
	float VHC = 1.0f;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, TC, VHC, 1.0f, 1.0f);
	int Ai = 0;
	int Bi = 0;
	float xi[3];
	xi[0] = -1;
	xi[1] = -1;
	xi[2] = -1;
	float output1 = simulator->calcMAB(xi, Ai, Bi);
	Eigen::MatrixXf Me(8, 8);
	for (int Ai = 0; Ai < 8; Ai++) {
		for (int Bi = 0; Bi < 8; Bi++) {
			Me(Ai, Bi) = simulator->integrate(&FEM_Simulator::calcMAB, 2, 0, Ai, Bi);
		}
	}
	// The truth values were calculated in matlab assuming Kint = 1 and deltaX = deltaY = deltaZ = 0.5
	EXPECT_TRUE((abs(1 / 64.0f) - output1) < 0.0001);
	for (int Ai = 0; Ai < 8; Ai++) {
		EXPECT_TRUE(((1 / 216.0f) - Me(Ai, Ai)) < 0.0001);
	}
	EXPECT_TRUE(((1 / 864.0f * TC) - Me(2, 0)) < 0.0001);
}

TEST(MainTest, TestCreateFjFunction1)
{
	std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
	float tissueSize[3] = { 1,1,1 };
	float TC = 1.0f;
	float VHC = 1.0f;
	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, TC, VHC, 1.0f, 1.0f);
	int Ai = 0;
	int Bi = 0;
	float xi[3];
	xi[0] = FEM_Simulator::A[Ai][0];
	xi[1] = FEM_Simulator::A[Ai][1];
	xi[2] = FEM_Simulator::A[Ai][2];
	float output1 = simulator->calcFqA(xi, Ai, 1);
	simulator->setFeQ();
	// The truth values were calculated in matlab assuming Kint = 1 and deltaX = deltaY = deltaZ = 0.5
	//EXPECT_TRUE((abs(1 / 64.0f) - output1) < 0.0001);
	//for (int Ai = 0; Ai < 8; Ai++) {
	//	EXPECT_TRUE(((1 / 216.0f) - Me(Ai, Ai)) < 0.0001);
	//}
	//EXPECT_TRUE(((1 / 864.0f * TC) - Me(2, 0)) < 0.0001);
}

TEST(MainTest, CompareLinearAndQuadratic1) {

	int nodeSize[3] = { 5,5,5 };
	std::vector<std::vector<std::vector<float>>> Temp(nodeSize[0], std::vector<std::vector<float>>(nodeSize[1], std::vector<float>(nodeSize[2])));
	std::vector<std::vector<std::vector<float>>> NFR(nodeSize[0], std::vector<std::vector<float>>(nodeSize[1], std::vector<float>(nodeSize[2])));
	srand(1);
	for (int i = 0; i < nodeSize[0]; i++) {
		for (int j = 0; j < nodeSize[1]; j++) {
			for (int k = 0; k < nodeSize[2]; k++) {
				Temp[i][j][k] = 0;
				NFR[i][j][k] = 1;
			}
		}
	}
	float tissueSize[3] = { 1,1,1 };
	float TC = 1.0f;
	int BC[6] = { 0,0,2,2,2,2 };
	FEM_Simulator* simulatorLin = new FEM_Simulator(Temp, tissueSize, TC, 1.0f, 1.0f, 1.0f, 2);
	FEM_Simulator* simulatorQuad = new FEM_Simulator(Temp, tissueSize, TC, 1.0f, 1.0f, 1.0f, 3);

	simulatorLin->deltaT = 0.05f;
	simulatorLin->tFinal = 1.0f;
	simulatorLin->setBoundaryConditions(BC);
	simulatorLin->setFlux(0);
	simulatorLin->setAmbientTemp(0);
	simulatorLin->setNFR(NFR);

	simulatorQuad->deltaT = 0.05f;
	simulatorQuad->tFinal = 1.0f;
	simulatorQuad->setBoundaryConditions(BC);
	simulatorQuad->setFlux(0);
	simulatorQuad->setAmbientTemp(0);
	simulatorQuad->setNFR(NFR);

	simulatorLin->createKMFelem();
	simulatorLin->performTimeStepping();
	simulatorQuad->createKMFelem();
	simulatorQuad->performTimeStepping();

	for (int k = 0; k < nodeSize[2]; k++) {
		for (int j = 0; j < nodeSize[1]; j++) {
			for (int i = 0; i < nodeSize[0]; i++) {
				int idx = i + j * nodeSize[0] + k * nodeSize[0] * nodeSize[1];
				EXPECT_TRUE(abs(simulatorQuad->Temp(idx) - simulatorQuad->Temp(idx)) < 0.001);
			}
		}
	}

}