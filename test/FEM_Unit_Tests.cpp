#include <gtest/gtest.h>
#include "FEM_Simulator.h"
#include <iostream>
#include <string>

class BaseSim : public testing::Test {
public:

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

		Eigen::VectorXf Temp = Eigen::VectorXf::Constant(27, 0.0f);
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

	static void SetUpTestSuite() {}

	static void TearDownTestSuite() {}
};

class SingleElem : public testing::Test {
public:

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
		mesh = Mesh::buildCubeMesh({ 2,2,1 }, { 2,2,2 }, { FLUX,FLUX,FLUX,FLUX,FLUX,FLUX });

		Eigen::VectorXf Temp = Eigen::VectorXf::Constant(8, 0.0f);
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

	static void SetUpTestSuite() {}

	static void TearDownTestSuite() {}
};

TEST_F(BaseSim, testSetFluenceRate) {

	std::array<float, 6> laserPose = { 0,0,-20,0,0,0 };
	float beamWaist = 0.0168;
	float wavelength = 10.6e-4;
	float laserPower = 1;
	femSim->setFluenceRate(laserPose, laserPower, beamWaist, wavelength);
	Eigen::VectorXf fluenceTrue(27); // matrices are transformed into vectors using column major
	fluenceTrue << 0.000000, 0.000017, 0.000000, 0.000017, 3.938834,
		0.000017, 0.000000, 0.000017, 0.000000, 0.000000, 0.000017,
		0.000000, 0.000017, 2.274098, 0.000017, 0.000000, 0.000017,
		0.000000, 0.000000, 0.000018, 0.000000, 0.000018, 1.314514,
		0.000018, 0.000000, 0.000018, 0.000000;

	int nNodes = 27;
	for (int i = 0; i < nNodes; i++) {
		//std::cout << "Simulator: " << femSimLin->FluenceRate(i) << " True Value: " << fluenceTrue(i) << std::endl;
		EXPECT_NEAR(femSim->fluenceRate()(i), fluenceTrue(i), 0.0001);
	}
}


TEST_F(BaseSim, testSetSensorTemps) {
	Eigen::VectorXf Temp = Eigen::VectorXf::Constant(27, 0.0f);
	Temp(4) = 20; // make the node at the center of the top surface 
	femSim->setTemp(Temp);
	// Check sensors at various positions in the tissue
	std::vector<std::array<float, 3>> sensorLocations = { {{0.0,0.0,0.0}, {0.5,0.5,0.25},{0.5,0.5,0.0},{0.5,0.0,0.0},
		{-0.5,0.0,0.0},{-0.5,0.5,0.0},{-0.5,0.5,0.5}} };
	femSim->setSensorLocations(sensorLocations);
	femSim->updateTemperatureSensors();

	std::vector<float> sTemps = femSim->sensorTemps();

	EXPECT_FLOAT_EQ(sTemps[0], 20.0f);
	EXPECT_FLOAT_EQ(sTemps[1], 20.0/8.0f);
	EXPECT_FLOAT_EQ(sTemps[2], 20.0 / 4.0f);
	EXPECT_FLOAT_EQ(sTemps[3], 20.0 / 2.0f);
	EXPECT_FLOAT_EQ(sTemps[4], 20.0 / 2.0f);
	EXPECT_FLOAT_EQ(sTemps[5], 20.0 / 4.0f);
	EXPECT_FLOAT_EQ(sTemps[6], 0.0);
}

TEST_F(SingleElem, testSetSensorLocation) {
	Eigen::VectorXf Temp = Eigen::VectorXf::Constant(8, 0.0f);
	Temp(4) = 20; // make the node at the center of the top surface 
	femSim->setTemp(Temp);
	// Check sensors at various positions in the tissue
	std::vector<std::array<float, 3>> sensorLocations = { {{0.0,0.0,0.0}, {0.0,0.0,0.05},{0.0,0.0,0.5},{0.0,0.0,0.95},
		{0.0,0.0,0.98}} };
	femSim->setSensorLocations(sensorLocations);
}