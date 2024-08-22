#include "pch.h"
#include "CppUnitTest.h"
#include "../HeatSimulation/FEM_Simulator.h"
#include "../HeatSimulation/FEM_Simulator.cpp"
#include <iostream>
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace FEMSimulatorTests
{
	TEST_CLASS(FEMSimulatorTests)
	{
	public:
		
		TEST_METHOD(TestCalculateNABase1)
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
					Assert::AreEqual(1.0f, output1);
					Assert::AreEqual(0.5f, output3);
					Assert::AreEqual(0.0f, output2);
				}
				else if (Ai == 1) {
					Assert::AreEqual(0.0f, output1);
					Assert::AreEqual(0.5f, output3);
					Assert::AreEqual(1.0f, output2);
				}
			}
		}

		TEST_METHOD(TestCalculateNABase2)
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
					Assert::AreEqual(1.0f, output1);
					Assert::AreEqual(0.375f, output4);
					Assert::AreEqual(0.0f, output2);
					Assert::AreEqual(0.0f, output3);
				}
				else if (Ai == 1) {
					Assert::AreEqual(0.0f, output1);
					Assert::AreEqual(0.75f, output4);
					Assert::AreEqual(1.0f, output2);
					Assert::AreEqual(0.0f, output3);
				}
				else if (Ai == 2) {
					Assert::AreEqual(0.0f, output1);
					Assert::AreEqual(-0.125f, output4);
					Assert::AreEqual(0.0f, output2);
					Assert::AreEqual(1.0f, output3);
				}
			}
		}

		/*TEST_METHOD(TestCalculateNA1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 1,1,1 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
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
			
		}*/

		/*TEST_METHOD(TestDetermineNodeFace1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 1,1,1 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);
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
		}*/

		TEST_METHOD(TestCalculateJ1)
		{
			std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} },
															   { {0,0,0}, {0,0,0}, {0,0,0} } };
			float tissueSize[3] = { 2,1,0.5 };
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);

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
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);

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
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);

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
			FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.006, 5, 1, 1.0f);

			Assert::AreEqual(tissueSize[0] / 4, simulator->Js3(0, 0));
			Assert::AreEqual(0.0f, simulator->Js3(0, 1));
			Assert::AreEqual(0.0f, simulator->Js3(1, 0));
			Assert::AreEqual(tissueSize[1] / 4, simulator->Js3(1, 1));
		}

		TEST_METHOD(TestInd2Sub1)
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
			Assert::AreEqual(0, sub[0]);
			Assert::AreEqual(0, sub[1]);
			Assert::AreEqual(0, sub[2]);
		}

		TEST_METHOD(TestInd2Sub2)
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
			Assert::AreEqual(0, sub[0]);
			Assert::AreEqual(1, sub[1]);
			Assert::AreEqual(0, sub[2]);
		}

		TEST_METHOD(TestInd2Sub3)
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
			Assert::AreEqual(0, sub[0]);
			Assert::AreEqual(0, sub[1]);
			Assert::AreEqual(1, sub[2]);
		}

		TEST_METHOD(TestInd2Sub4)
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
			Assert::AreEqual(1, sub[0]);
			Assert::AreEqual(2, sub[1]);
			Assert::AreEqual(5, sub[2]);
		}

		TEST_METHOD(TestCreateKABFunction1)
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

		TEST_METHOD(TestCreateMABFunction1)
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
			float output1 = simulator->createMABFunction(xi, Ai, Bi);
			Eigen::MatrixXf Me(8, 8);
			for (int Ai = 0; Ai < 8; Ai++) {
				for (int Bi = 0; Bi < 8; Bi++) {
					Me(Ai, Bi) = simulator->integrate(&FEM_Simulator::createMABFunction, 2, 0, Ai, Bi);
				}
			}
			// The truth values were calculated in matlab assuming K = 1 and deltaX = deltaY = deltaZ = 0.5
			Assert::IsTrue((abs(1 / 64.0f) - output1) < 0.0001);
			for (int Ai = 0; Ai < 8; Ai++) {
				Assert::IsTrue(((1 / 216.0f) - Me(Ai, Ai)) < 0.0001);
			}
			Assert::IsTrue(((1 / 864.0f * TC) - Me(2, 0)) < 0.0001);
		}

		TEST_METHOD(TestCreateFjFunction1)
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
			float output1 = simulator->createFjFunction(xi, Ai, 1);
			simulator->setFj();
			// The truth values were calculated in matlab assuming K = 1 and deltaX = deltaY = deltaZ = 0.5
			//Assert::IsTrue((abs(1 / 64.0f) - output1) < 0.0001);
			//for (int Ai = 0; Ai < 8; Ai++) {
			//	Assert::IsTrue(((1 / 216.0f) - Me(Ai, Ai)) < 0.0001);
			//}
			//Assert::IsTrue(((1 / 864.0f * TC) - Me(2, 0)) < 0.0001);
		}

		//TEST_METHOD(TestCreateKMF1) {
		//	//checking that createKMF and createKMFelem create the same matrices.
		//	int nodeSize[3] = { 10,10,10 };
		//	std::vector<std::vector<std::vector<float>>> Temp(nodeSize[0], std::vector<std::vector<float>>(nodeSize[1], std::vector<float>(nodeSize[2])));
		//	std::vector<std::vector<std::vector<float>>> NFR(nodeSize[0], std::vector<std::vector<float>>(nodeSize[1], std::vector<float>(nodeSize[2])));
		//	for (int i = 0; i < nodeSize[0]; i++) {
		//		for (int j = 0; j < nodeSize[1]; j++) {
		//			for (int k = 0; k < nodeSize[2]; k++) {
		//				Temp[i][j][k] = 0.0f;
		//				NFR[i][j][k] = 1.5f;
		//			}
		//		}
		//	}
		//	float tissueSize[3] = { 1.0f,1.0f,1.0f };
		//	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.0062, 5.22, 100, 1);
		//	simulator->deltaT = 0.1f;
		//	simulator->tFinal = 1.0f;
		//	int BC[6] = { 1,0,1,0,2,2 };
		//	simulator->setBoundaryConditions(BC);
		//	simulator->setJn(1);
		//	simulator->setAmbientTemp(0);
		//	simulator->setNFR(NFR);

		//	simulator->createKMFelem();
		//	Eigen::VectorXf Felem = simulator->F;
		//	Eigen::SparseMatrix<float> Kelem = simulator->K;
		//	Eigen::SparseMatrix<float> Melem = simulator->M;
		//	simulator->createKMF();
		//	Eigen::VectorXf Fnode = simulator->F;
		//	Eigen::SparseMatrix<float> Knode = simulator->K;
		//	Eigen::SparseMatrix<float> Mnode = simulator->M;
		//	int totalNodes = nodeSize[0] * nodeSize[1] * nodeSize[2] - simulator->dirichletNodes.size();
		//	for (int i = 0; i < totalNodes; i++) {
		//		Assert::IsTrue(abs(Fnode(i) - Felem(i)) < 0.0000001, (std::wstring(L"F - Error on index i: ") + std::to_wstring(i)).c_str());
		//		for (int j = 0; j < totalNodes; j++) {
		//			Assert::IsTrue(abs(Knode.coeffRef(i, j) - Kelem.coeffRef(i,j)) < 0.0000001, (std::wstring(L"K - Error on index i: ") + std::to_wstring(i) + L", j: " + std::to_wstring(j)).c_str());
		//			Assert::IsTrue(abs(Mnode.coeffRef(i, j) - Melem.coeffRef(i, j)) < 0.0000001, (std::wstring(L"M - sError on index i: ") + std::to_wstring(i) + L", j: " + std::to_wstring(j)).c_str());
		//		}
		//	}

		//	simulator->performTimeStepping();
		//}

		//TEST_METHOD(TestCreateKMF2) {
		//	//checking that createKMF and createKMFelem create the same matrices for elemental NFR.
		//	int nodeSize[3] = { 10,10,10 };
		//	std::vector<std::vector<std::vector<float>>> Temp(nodeSize[0], std::vector<std::vector<float>>(nodeSize[1], std::vector<float>(nodeSize[2])));
		//	std::vector<std::vector<std::vector<float>>> NFR(nodeSize[0]-1, std::vector<std::vector<float>>(nodeSize[1]-1, std::vector<float>(nodeSize[2]-1)));
		//	for (int i = 0; i < nodeSize[0]; i++) {
		//		for (int j = 0; j < nodeSize[1]; j++) {
		//			for (int k = 0; k < nodeSize[2]; k++) {
		//				Temp[i][j][k] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) + 1;
		//			}
		//		}
		//	}
		//	// setting up elemental NFR
		//	for (int i = 0; i < nodeSize[0]-1; i++) {
		//		for (int j = 0; j < nodeSize[1]-1; j++) {
		//			for (int k = 0; k < nodeSize[2]-1; k++) {
		//				NFR[i][j][k] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) + 1;
		//			}
		//		}
		//	}
		//	float tissueSize[3] = { 1.0f,1.0f,1.0f };
		//	FEM_Simulator* simulator = new FEM_Simulator(Temp, tissueSize, 0.0062, 5.22, 100, 1);
		//	simulator->deltaT = 0.1f;
		//	simulator->tFinal = 1.0f;
		//	int BC[6] = { 1,0,1,0,2,2 };
		//	simulator->setBoundaryConditions(BC);
		//	simulator->setJn(1);
		//	simulator->setAmbientTemp(0);
		//	simulator->setNFR(NFR);

		//	simulator->createKMFelem();
		//	Eigen::VectorXf Felem = simulator->F;
		//	Eigen::SparseMatrix<float> Kelem = simulator->K;
		//	Eigen::SparseMatrix<float> Melem = simulator->M;
		//	simulator->createKMF();
		//	Eigen::VectorXf Fnode = simulator->F;
		//	Eigen::SparseMatrix<float> Knode = simulator->K;
		//	Eigen::SparseMatrix<float> Mnode = simulator->M;
		//	int totalNodes = nodeSize[0] * nodeSize[1] * nodeSize[2] - simulator->dirichletNodes.size();
		//	for (int i = 0; i < totalNodes; i++) {
		//		Assert::IsTrue(abs(Fnode(i) - Felem(i)) < 0.0000001, (std::wstring(L"F - Error on index i: ") + std::to_wstring(i)).c_str());
		//		for (int j = 0; j < totalNodes; j++) {
		//			Assert::IsTrue(abs(Knode.coeffRef(i, j) - Kelem.coeffRef(i, j)) < 0.0000001, (std::wstring(L"K - Error on index i: ") + std::to_wstring(i) + L", j: " + std::to_wstring(j)).c_str());
		//			Assert::IsTrue(abs(Mnode.coeffRef(i, j) - Melem.coeffRef(i, j)) < 0.0000001, (std::wstring(L"M - sError on index i: ") + std::to_wstring(i) + L", j: " + std::to_wstring(j)).c_str());
		//		}
		//	}

		//	simulator->performTimeStepping();
		//}
	};
}
