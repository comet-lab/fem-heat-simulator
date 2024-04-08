#include "pch.h"
#include "CppUnitTest.h"
#include "../HeatSimulation/FEM_Simulator.h"
#include "../HeatSimulation/FEM_Simulator.cpp"
#include <iostream>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace FEMSimulatorTests
{
	TEST_CLASS(FEMSimulatorTests)
	{
	public:
		
		TEST_METHOD(TestCalculateNA1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 1,1,1 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);
			float xi1[3], xi2[3];
			for (int Ai = 0; Ai < 8; Ai++) {
				xi1[0] = FEM_Simulator::A[Ai][0];
				xi1[1] = FEM_Simulator::A[Ai][1];
				xi1[2] = FEM_Simulator::A[Ai][2];
				xi2[0] = FEM_Simulator::A[(Ai + 1) % 8][0];
				xi2[1] = FEM_Simulator::A[(Ai + 1) % 8][1];
				xi2[2] = FEM_Simulator::A[(Ai + 1) % 8][2];
				float output1 = simulator->calculateNA(xi1, Ai);
				float output2 = simulator->calculateNA(xi2, Ai);

				Assert::AreEqual(1.0f, output1);
				Assert::AreEqual(0.0f, output2);				
			}
			
		}
		TEST_METHOD(TestDetermineNodeFace1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 1,1,1 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);
			int faces[27];
			for (int i = 0; i < 27; i++) {
				faces[i] = simulator->determineNodeFace(i);
			}

			int expected0 = FEM_Simulator::tissueFace::TOP + FEM_Simulator::tissueFace::FRONT + FEM_Simulator::tissueFace::LEFT;
			int expected2 = FEM_Simulator::tissueFace::TOP + FEM_Simulator::tissueFace::FRONT + FEM_Simulator::tissueFace::RIGHT;
			int expected13 = FEM_Simulator::tissueFace::INTERNAL;
			int expected16 = FEM_Simulator::tissueFace::BACK;
			int expected22 = FEM_Simulator::tissueFace::BOTTOM;
			Assert::AreEqual(expected0, faces[0]);
			Assert::AreEqual(expected2, faces[2]);
			Assert::AreEqual(expected13, faces[13]);
			Assert::AreEqual(expected16, faces[16]);
			Assert::AreEqual(expected22, faces[22]);
		}

		TEST_METHOD(TestCalculateJ1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);

			Assert::AreEqual(tissueSize[0] / 4, simulator->J(0, 0));
			Assert::AreEqual(0.0f, simulator->J(0, 1));
			Assert::AreEqual(0.0f, simulator->J(0, 2));
			Assert::AreEqual(0.0f, simulator->J(1, 0));
			Assert::AreEqual(tissueSize[1] / 4, simulator->J(1, 1));
			Assert::AreEqual(0.0f, simulator->J(1, 2));
			Assert::AreEqual(0.0f, simulator->J(2, 0));
			Assert::AreEqual(0.0f, simulator->J(2, 1));
			Assert::AreEqual(tissueSize[2] / 4, simulator->J(2, 2));
		}

		TEST_METHOD(TestCalculateJs1_1)
		{
			// Js1 is the partial jacobian for the front/back faces. aka the y-z plane
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);

			Assert::AreEqual(tissueSize[1] / 4, simulator->Js1(0, 0));
			Assert::AreEqual(0.0f, simulator->Js1(0, 1));
			Assert::AreEqual(0.0f, simulator->Js1(1, 0));
			Assert::AreEqual(tissueSize[2] / 4, simulator->Js1(1, 1));
		}

		TEST_METHOD(TestCalculateJs2_1)
		{
			// Js2 is the partial jacobian for the left/right faces. aka the x-z plane
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);

			Assert::AreEqual(tissueSize[0] / 4, simulator->Js2(0, 0));
			Assert::AreEqual(0.0f, simulator->Js2(0, 1));
			Assert::AreEqual(0.0f, simulator->Js2(1, 0));
			Assert::AreEqual(tissueSize[2] / 4, simulator->Js2(1, 1));
		}

		TEST_METHOD(TestCalculateJs3_1)
		{
			// Js3 is the partial jacobian for the top/bottom faces. aka the x-y plane
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);

			Assert::AreEqual(tissueSize[0] / 4, simulator->Js3(0, 0));
			Assert::AreEqual(0.0f, simulator->Js3(0, 1));
			Assert::AreEqual(0.0f, simulator->Js3(1, 0));
			Assert::AreEqual(tissueSize[1] / 4, simulator->Js3(1, 1));
		}

		TEST_METHOD(TestCalculateNA_Xi1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);
			float xi[3];
			for (int Ai = 0; Ai < 1; Ai++) {
				xi[0] = FEM_Simulator::A[Ai][0];
				xi[1] = FEM_Simulator::A[Ai][1];
				xi[2] = FEM_Simulator::A[Ai][2];
				float output1 = simulator->calculateNA_xi(xi, Ai); // test at Ai
				xi[0] = -xi[0];
				float output2 = simulator->calculateNA_xi(xi, Ai); // test but flip x
				xi[0] = -xi[0]; // flip x back
				xi[1] = -xi[1];
				float output3 = simulator->calculateNA_xi(xi, Ai); // test but flip y
				xi[1] = -xi[1]; // flip y back
				xi[2] = -xi[2]; 
				float output4 = simulator->calculateNA_xi(xi, Ai); // test but flip z

				Assert::AreEqual(float(1 / 2.0f * FEM_Simulator::A[Ai][0]), output1);
				Assert::AreEqual(float(1 / 2.0f * FEM_Simulator::A[Ai][0]), output2);
				Assert::AreEqual(0.0f, output3);
				Assert::AreEqual(0.0f, output4);
			}
		}

		TEST_METHOD(TestCalculateNA_Eta1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);
			float xi[3];
			for (int Ai = 0; Ai < 1; Ai++) {
				xi[0] = FEM_Simulator::A[Ai][0];
				xi[1] = FEM_Simulator::A[Ai][1];
				xi[2] = FEM_Simulator::A[Ai][2];
				float output1 = simulator->calculateNA_eta(xi, Ai); // test at Ai
				xi[0] = -xi[0];
				float output2 = simulator->calculateNA_eta(xi, Ai); // test but flip x
				xi[0] = -xi[0]; // flip x back
				xi[1] = -xi[1];
				float output3 = simulator->calculateNA_eta(xi, Ai); // test but flip y
				xi[1] = -xi[1]; // flip y back
				xi[2] = -xi[2];
				float output4 = simulator->calculateNA_eta(xi, Ai); // test but flip z

				Assert::AreEqual(float(1 / 2.0f * FEM_Simulator::A[Ai][0]), output1);
				Assert::AreEqual(0.0f, output2);
				Assert::AreEqual(float(1 / 2.0f * FEM_Simulator::A[Ai][0]), output3);
				Assert::AreEqual(0.0f, output4);
			}
		}


		TEST_METHOD(TestCalculateNA_Zeta1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);
			float xi[3];
			for (int Ai = 0; Ai < 1; Ai++) {
				xi[0] = FEM_Simulator::A[Ai][0];
				xi[1] = FEM_Simulator::A[Ai][1];
				xi[2] = FEM_Simulator::A[Ai][2];
				float output1 = simulator->calculateNA_zeta(xi, Ai); // test at Ai
				xi[0] = -xi[0];
				float output2 = simulator->calculateNA_zeta(xi, Ai); // test but flip x
				xi[0] = -xi[0]; // flip x back
				xi[1] = -xi[1];
				float output3 = simulator->calculateNA_zeta(xi, Ai); // test but flip y
				xi[1] = -xi[1]; // flip y back
				xi[2] = -xi[2];
				float output4 = simulator->calculateNA_zeta(xi, Ai); // test but flip z

				Assert::AreEqual(float(1 / 2.0f * FEM_Simulator::A[Ai][0]), output1);
				Assert::AreEqual(0.0f, output2);
				Assert::AreEqual(0.0f, output3);
				Assert::AreEqual(float(1 / 2.0f * FEM_Simulator::A[Ai][0]), output4);
			}
		}

		TEST_METHOD(TestInd2Sub1)
		{
			int sub[3];
			int index = 0;
			int size[3] = { 10,10,10 };
			FEM_Simulator::ind2sub(index, size, sub);
			Assert::AreEqual(0, sub[0]);
			Assert::AreEqual(0, sub[1]);
			Assert::AreEqual(0, sub[2]);
		}

		TEST_METHOD(TestInd2Sub2)
		{
			int sub[3];
			int index = 10;
			int size[3] = { 10,10,10 };
			FEM_Simulator::ind2sub(index, size, sub);
			Assert::AreEqual(0, sub[0]);
			Assert::AreEqual(1, sub[1]);
			Assert::AreEqual(0, sub[2]);
		}

		TEST_METHOD(TestInd2Sub3)
		{
			int sub[3];
			int index = 100;
			int size[3] = { 10,10,10 };
			FEM_Simulator::ind2sub(index, size, sub);
			Assert::AreEqual(0, sub[0]);
			Assert::AreEqual(0, sub[1]);
			Assert::AreEqual(1, sub[2]);
		}

		TEST_METHOD(TestInd2Sub4)
		{
			int sub[3];
			int index = 521;
			int size[3] = { 10,10,10 };
			FEM_Simulator::ind2sub(index, size, sub);
			Assert::AreEqual(1, sub[0]);
			Assert::AreEqual(2, sub[1]);
			Assert::AreEqual(5, sub[2]);
		}

		TEST_METHOD(TestGetGlobalNodes1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);
			int e = 0;
			int elementGlobalNodes[8]; //global nodes for element e
			simulator->getGlobalNodesFromElem(e, elementGlobalNodes);
			// Note that these tests will be wrong if the order of A changes in the .cpp file. 
			Assert::AreEqual(0,elementGlobalNodes[0]);
			Assert::AreEqual(1, elementGlobalNodes[1]);
			Assert::AreEqual(4, elementGlobalNodes[2]);
			Assert::AreEqual(3, elementGlobalNodes[3]);
			Assert::AreEqual(9, elementGlobalNodes[4]);
			Assert::AreEqual(10, elementGlobalNodes[5]);
			Assert::AreEqual(13, elementGlobalNodes[6]);
			Assert::AreEqual(12, elementGlobalNodes[7]);

		}

		TEST_METHOD(TestGetGlobalNodes2)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);
			int e = 5;
			int elementGlobalNodes[8]; //global nodes for element e
			simulator->getGlobalNodesFromElem(e, elementGlobalNodes);
			// Note that these tests will be wrong if the order of A changes in the .cpp file. 
			Assert::AreEqual(10, elementGlobalNodes[0]);
			Assert::AreEqual(11, elementGlobalNodes[1]);
			Assert::AreEqual(14, elementGlobalNodes[2]);
			Assert::AreEqual(13, elementGlobalNodes[3]);
			Assert::AreEqual(19, elementGlobalNodes[4]);
			Assert::AreEqual(20, elementGlobalNodes[5]);
			Assert::AreEqual(23, elementGlobalNodes[6]);
			Assert::AreEqual(22, elementGlobalNodes[7]);

		}

		TEST_METHOD(TestGetGlobalNodes3)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1);
			int e = 7;
			int elementGlobalNodes[8]; //global nodes for element e
			simulator->getGlobalNodesFromElem(e, elementGlobalNodes);
			// Note that these tests will be wrong if the order of A changes in the .cpp file. 
			Assert::AreEqual(13, elementGlobalNodes[0]);
			Assert::AreEqual(14, elementGlobalNodes[1]);
			Assert::AreEqual(17, elementGlobalNodes[2]);
			Assert::AreEqual(16, elementGlobalNodes[3]);
			Assert::AreEqual(22, elementGlobalNodes[4]);
			Assert::AreEqual(23, elementGlobalNodes[5]);
			Assert::AreEqual(26, elementGlobalNodes[6]);
			Assert::AreEqual(25, elementGlobalNodes[7]);

		}

		TEST_METHOD(TestCreateKABFunction1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 1,1,1 };
			float TC = 1.0f; 
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, TC, 1.0f, 1.0f);
			int Ai = 0;
			int Bi = 0;
			float xi[3];
			xi[0] = FEM_Simulator::A[Ai][0];
			xi[1] = FEM_Simulator::A[Ai][1];
			xi[2] = FEM_Simulator::A[Ai][2];
			float output1 = simulator->createKABFunction(xi, Ai, Bi);
			Eigen::MatrixXf Ke(8, 8);
			for (int Ai = 0; Ai < 8; Ai++) {
				for (int Bi = 0; Bi < 8; Bi++) {
					Ke(Ai, Bi) = simulator->integrate(&FEM_Simulator::createKABFunction, 2, 0, Ai, Bi);
				}
			}
			// The truth values were calculated in matlab assuming K = 1 and deltaX = deltaY = deltaZ = 0.5
			Assert::IsTrue((abs(3 * TC / 16.0f) - output1) < 0.0001);
			for (int Ai = 0; Ai < 8; Ai++) {
				Assert::IsTrue(((1 / 6.0f * TC) - Ke(Ai, Ai)) < 0.0001);
			}
			Assert::IsTrue(((-1 / 24.0f * TC) - Ke(2, 0)) < 0.0001);
		}

		TEST_METHOD(TestCreateKABFunction1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 1,1,1 };
			float TC = 1.0f;
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, TC, 1.0f, 1.0f);
			int Ai = 0;
			int Bi = 0;
			float xi[3];
			xi[0] = FEM_Simulator::A[Ai][0];
			xi[1] = FEM_Simulator::A[Ai][1];
			xi[2] = FEM_Simulator::A[Ai][2];
			float output1 = simulator->createKABFunction(xi, Ai, Bi);
			Eigen::MatrixXf Ke(8, 8);
			for (int Ai = 0; Ai < 8; Ai++) {
				for (int Bi = 0; Bi < 8; Bi++) {
					Ke(Ai, Bi) = simulator->integrate(&FEM_Simulator::createKABFunction, 2, 0, Ai, Bi);
				}
			}
			// The truth values were calculated in matlab assuming K = 1 and deltaX = deltaY = deltaZ = 0.5
			Assert::IsTrue((abs(3 * TC / 16.0f) - output1) < 0.0001);
			for (int Ai = 0; Ai < 8; Ai++) {
				Assert::IsTrue(((1 / 6.0f * TC) - Ke(Ai, Ai)) < 0.0001);
			}
			Assert::IsTrue(((-1 / 24.0f * TC) - Ke(2, 0)) < 0.0001);
		}
	};
}
