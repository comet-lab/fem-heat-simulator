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

class FullSim : public testing::Test {
public:

	FEM_Simulator* sim = nullptr;
	std::vector<Node> nodes;
	Element elem;
	std::vector<BoundaryFace> boundaryFaces;

	Mesh mesh;

	void SetUp() override {
		std::array<float, 3> tissueSize = { 2.0f,2.0f,1.0f };
		std::array<long, 3> nodesPerAxis = { 81,81,51 };
		long nNodes = nodesPerAxis[0] * nodesPerAxis[1] * nodesPerAxis[2];
		long nElems = (nodesPerAxis[0] - 1) * (nodesPerAxis[1] - 1) * (nodesPerAxis[2] - 1);
		//std::array<BoundaryType,6> BC = { CONVECTION,CONVECTION,CONVECTION ,CONVECTION ,CONVECTION ,CONVECTION };
		std::array<BoundaryType, 6> BC = { HEATSINK,HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK};
		mesh = Mesh::buildCubeMesh(tissueSize, nodesPerAxis, BC);

		float initialTemp = 20.0f;
		Eigen::VectorXf Temp = Eigen::VectorXf::Constant(nNodes, initialTemp);
		Eigen::VectorXf FluenceRate = Eigen::VectorXf::Constant(nNodes, 0);

		srand(1);

		float mua = 200;
		float tc = 0.0062;
		float vhc = 4.3;
		float htc = 0.01;
		sim = new FEM_Simulator(mua, vhc, tc, htc);
		sim->setMesh(mesh);
		sim->setTemp(Temp);
		sim->setAlpha(1);
		sim->setDt(0.05f);
		sim->setHeatFlux(0.0);
		sim->setAmbientTemp(24.0f);
		sim->silentMode = false;
		sim->enableGPU();
		std::array<float, 6> laserPose = { 0.0f,0,-35,0,0,0 };
		sim->setFluenceRate(laserPose, 0, 0.0168, 10.6e-4);
		sim->buildMatrices();
		sim->silentMode = true;
	}

	// ~QueueTest() override = default;
	void TearDown() override {
		std::cout << "Teardown" << std::endl;
		delete sim;
		sim = nullptr;
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


TEST_F(SingleElem, copyConstructorTest) {

	FEM_Simulator* simCopy = new FEM_Simulator(*femSim);
	Eigen::VectorXf temp2 = Eigen::VectorXf::Constant(8, 1.0f);;
	simCopy->setTemp(temp2);
	simCopy->setMUA(0);
	simCopy->setTC(0);
	simCopy->setVHC(0);
	simCopy->setDt(10);
	simCopy->setAlpha(0);

	auto sim1Temp = femSim->Temp();
	auto sim2Temp = simCopy->Temp();
	for (int i = 0; i < 8; i++)
	{
		ASSERT_NE(sim1Temp(i), sim2Temp(i));
	}

	ASSERT_NE(simCopy->MUA(), femSim->MUA());
	ASSERT_NE(simCopy->TC(), femSim->TC());
	ASSERT_NE(simCopy->VHC(), femSim->VHC());
	ASSERT_FLOAT_EQ(simCopy->HTC(), femSim->HTC());
	ASSERT_NE(simCopy->dt(), femSim->dt());
	ASSERT_NE(simCopy->alpha(), femSim->alpha());

}


TEST_F(FullSim, parLoopTest) {

	const int numThreads = 5;
	omp_set_num_threads(numThreads);
	int totalSims = 10;
	int numRuns = 5;
	std::vector<FEM_Simulator*> simObjs(numThreads);;
	for (int i = 0; i < numThreads; ++i)
	{
		simObjs[i] = new FEM_Simulator(*sim);
		simObjs[i]->initializeTimeIntegration(); // only initialize time integration here
	}
	Eigen::MatrixXf TempStorage(mesh.nodes().size(), totalSims);

	for (int r = 0; r < numRuns; r++)
	{

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < totalSims; i++)
		{
			int tid = omp_get_thread_num();
			FEM_Simulator* currSim = simObjs[tid];
			Eigen::VectorXf newT = Eigen::VectorXf::Constant(mesh.nodes().size(), i);
			currSim->setTemp(newT);
			currSim->singleStep();
			TempStorage.col(i) = currSim->Temp();
		}
	}
	
	// Dirichlet boundaries and no laser input means that each column should have values equal to the column number
	for (int i = 0; i < totalSims; i++)
		ASSERT_FLOAT_EQ(TempStorage(0, i), i);
}