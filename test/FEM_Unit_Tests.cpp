#include <gtest/gtest.h>
#include "FEM_Simulator.h"
#include <iostream>
#include <string>

class BaseSim : public testing::Test {
protected:

	FEM_Simulator* femSim = nullptr;
	std::vector<Node> nodes;
	Element elem;
	std::vector<BoundaryFace> boundaryFaces;
	
	Mesh mesh;
	float mua = 1.0f;
	float tc = 1.0f;
	float vhc = 1.0f;
	float htc = 1.0f;

	void SetUp() override {
		// q0_ remains empty
		mesh = Mesh::buildCubeMesh({ 2,2,1 }, { 3,3,3 }, { FLUX,FLUX,FLUX,FLUX,FLUX,FLUX });

		std::vector<std::vector<std::vector<float>>> Temp = { { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} },
													   { {0,0,0}, {0,0,0}, {0,0,0} } };
		float tissueSize[3] = { 1,1,1 };	

		femSim = new FEM_Simulator(mua, vhc, tc, htc);
		femSim->setMesh(mesh);
		femSim->setTemp(Temp);
		femSim->buildMatrices();
	}

	// ~QueueTest() override = default;
	void TearDown() override {
		std::cout << "Teardown" << std::endl;
		delete femSim;
		femSim = nullptr;
	}



	static void SetUpTestSuite() {

	}

	static void TearDownTestSuite() {

	}
};

TEST_F(BaseSim, testCopyConstructor) {

	FEM_Simulator* femSim = new FEM_Simulator(*femSim);
}

TEST_F(BaseSim, CompareLinearAndQuadratic1) {

	//int nodesPerAxis[3] = { 5,5,5 };
	//std::vector<std::vector<std::vector<float>>> Temp(nodesPerAxis[0], std::vector<std::vector<float>>(nodesPerAxis[1], std::vector<float>(nodesPerAxis[2])));
	//std::vector<std::vector<std::vector<float>>> FluenceRate(nodesPerAxis[0], std::vector<std::vector<float>>(nodesPerAxis[1], std::vector<float>(nodesPerAxis[2])));
	//srand(1);
	//for (int i = 0; i < nodesPerAxis[0]; i++) {
	//	for (int j = 0; j < nodesPerAxis[1]; j++) {
	//		for (int k = 0; k < nodesPerAxis[2]; k++) {
	//			Temp[i][j][k] = 0;
	//			FluenceRate[i][j][k] = 1;
	//		}
	//	}
	//}

	//float tissueSize[3] = { 1,1,1 };
	//float TC = 1.0f;
	//int BC[6] = { 0,0,2,2,2,2 };

	//// already defined in BaseSim so we need to clear and reinstatiate
	//delete femSimLin;
	//femSimLin = nullptr;
	//delete femSimQuad;
	//femSimQuad = nullptr;

	//femSimLin = new FEM_Simulator(Temp, tissueSize, TC, 1.0f, 1.0f, 1.0f, 2);
	//femSimQuad = new FEM_Simulator(Temp, tissueSize, TC, 1.0f, 1.0f, 1.0f, 3);

	//femSimLin->deltaT_ = 0.05f;
	//femSimLin->setBoundaryConditions(BC);
	//femSimLin->setHeatFlux(0);
	//femSimLin->setAmbientTemp(0);
	//femSimLin->setFluenceRate(FluenceRate);

	//femSimQuad->deltaT_ = 0.05f;
	//femSimQuad->setBoundaryConditions(BC);
	//femSimQuad->setHeatFlux(0);
	//femSimQuad->setAmbientTemp(0);
	//femSimQuad->setFluenceRate(FluenceRate);

	//femSimLin->initializeModel();
	//femSimLin->multiStep(1.0f);
	//femSimQuad->initializeModel();
	//femSimQuad->multiStep(1.0f);

	//for (int k = 0; k < nodesPerAxis[2]; k++) {
	//	for (int j = 0; j < nodesPerAxis[1]; j++) {
	//		for (int i = 0; i < nodesPerAxis[0]; i++) {
	//			int idx = i + j * nodesPerAxis[0] + k * nodesPerAxis[0] * nodesPerAxis[1];
	//			EXPECT_TRUE(abs(femSimQuad->Temp_(idx) - femSimQuad->Temp_(idx)) < 0.001);
	//		}
	//	}
	//}

}

TEST_F(BaseSim, testPositionToElement1) {
	
}

TEST_F(BaseSim, testSetFluenceRate) {

	std::array<float,6> laserPose = { 0,0,-20,0,0,0 };
	float beamWaist = 0.0168;
	float laserPower = 1; 
	femSim->setFluenceRate(laserPose, laserPower, beamWaist);
	Eigen::VectorXf fluenceTrue(27); // matrices are transformed into vectors using column major
	fluenceTrue << 0.008097328938, 0.1785890066, 0.008097328938, 0.1785890066, 3.938833843, 0.1785890066, 0.008097328938, 0.1785890066, 0.008097328938,
		0.007903577769, 0.1716547562, 0.007903577769, 0.1716547562, 3.728103424, 0.1716547562, 0.007903577769, 0.1716547562, 0.007903577769,
		0.004799076444, 0.07942576033, 0.004799076444, 0.07942576033, 1.314513631, 0.07942576033, 0.004799076444, 0.07942576033, 0.004799076444;

	int nNodes = 27;
	for (int i = 0; i < nNodes; i++) {
		//std::cout << "Simulator: " << femSimLin->FluenceRate(i) << " True Value: " << fluenceTrue(i) << std::endl;
		ASSERT_TRUE(abs(femSim->fluenceRate()(i) - fluenceTrue(i)) < 0.000001);
	}
}


TEST(SecondaryTest, testSetSensorTemps) {

	//// Multi Layer Test
	//int nodesPerAxis[3] = { 21,21,20 };
	//std::vector<std::vector<std::vector<float>>> Temp(nodesPerAxis[0], std::vector<std::vector<float>>(nodesPerAxis[1], std::vector<float>(nodesPerAxis[2])));
	//for (int i = 0; i < nodesPerAxis[0]; i++) {
	//	for (int j = 0; j < nodesPerAxis[1]; j++) {
	//		for (int k = 0; k < nodesPerAxis[2]; k++)
	//			Temp[i][j][k] = i + j * nodesPerAxis[0] + k * (nodesPerAxis[0] * nodesPerAxis[1]);
	//	}
	//}
	//float layerHeight = 0.1;
	//int elemsInLayer = 10;
	//float tissueSize[3] = { 2,2,1 };
	//int Nn1d = 2;
	//float mua = 1.0f;
	//float tc = 1.0f;
	//float vhc = 1.0f;
	//float htc = 1.0f;

	//FEM_Simulator* femSimLin = new FEM_Simulator(Temp, tissueSize, tc, vhc, mua, htc, Nn1d);
	//femSimLin->setLayer(layerHeight, elemsInLayer);
	//femSimLin->initializeBoundaryNodes();

	//int nNodes = nodesPerAxis[0] * nodesPerAxis[1] * nodesPerAxis[2];

	//std::vector<std::array<float, 3>> sensorLocations = { 
	//	{ -1, -1, 0 },
	//	{ 0, 0, 0 },
	//	{ -0.25, -0.25, 0.02 },
	//	{ 1, 1, 1 },
	//	{ -0.25, -0.25, 0.15 },
	//	{ -0.25, 0.25, 0.5 },
	//	{ 0.25, -0.25, 0.045 }
	//	};

	//float laserPose[6] = { 0,0,0,0,0,0 };
	//femSimLin->setFluenceRate(laserPose, 0, 0.1);
	//femSimLin->initializeModel();
	//femSimLin->setSensorLocations(sensorLocations);
	//femSimLin->initializeSensorTemps(1.0f);
	//femSimLin->updateTemperatureSensors(0);

	//std::vector<float> expectedSensorTemp = {0,220,1047,8819,4795.5,6444,2154.5};

	//for (int i = 0; i < 7; i++) {
	//	ASSERT_FLOAT_EQ(expectedSensorTemp[i], femSimLin->sensorTemps_[i][0]);
	//}

	//delete femSimLin;
	//femSimLin = nullptr;
}