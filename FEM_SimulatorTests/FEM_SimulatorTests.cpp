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
			float xi[3];
			for (int Ai = 0; Ai < 8; Ai++) {
				xi[0] = FEM_Simulator::A[Ai][0];
				xi[1] = FEM_Simulator::A[Ai][1];
				xi[2] = FEM_Simulator::A[Ai][2];
				float output1 = simulator->calculateNA(xi, Ai);
				float output2 = simulator->calculateNA(xi, (Ai+1)%8);

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

		}
	};
}
