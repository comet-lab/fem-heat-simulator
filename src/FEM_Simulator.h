#pragma once
#include <vector> 
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <stdexcept>

class FEM_Simulator
{
public:
	bool silentMode = false;

	struct element {             // Structure declaration
		int elementNumber;         // Member (int variable)
		float globalNodePositions[8][3];   // Member (string variable)
	};       // Structure variable
	static const int A[8][3];

	// One node can belong to multiple faces so we use binary to set the face value
	// 0 - internal node, 1 - top face, 2 - bottom face, 4 - front face, 8 - right fce, 16 - back face, 32 - left face
	enum tissueFace { INTERNAL = 0, TOP = 1, BOTTOM = 2, FRONT = 4, RIGHT = 8, BACK = 16, LEFT = 32 };
	// This maps the face to the axis number and direction of the face: top, bottom, front, right, back, left
	// The values listed here are heavily tied to our reference frame decisions and the function @determineNodeFace
	const int dimMap[6] = { -3, 3, 1, 2, -1, -2 };
	// -3: We are on the top face    - surface normal is in -z direction
	//  3: We are on the bottom face - surface normal is in +z direction
	//  1: We are on the front face  - surface normal is +x direction
	//  2: We are on the right face  - surface normal is +y direction
	// -1: We are on the back face   - surface normal is -x direction
	// -2: We are on the left face   - surface normal is -y direction

	// This maps the face on an element to the local node numbers on that face: top,bot,front,right,back,left
	std::array<std::vector<int>,6> elemNodeSurfaceMap;
	//0 - heat sink, 1 - flux boundary, 2 - convection boundary
	enum boundaryCond { HEATSINK, FLUX, CONVECTION };

	int gridSize[3] = { 1,1,1 }; // Number of elements in x, y, and z [voxels]
	int nodeSize[3] = { 2,2,2 }; // Number of nodes in x, y, and z. Should be gridSize + 1;
	float tissueSize[3] = { 1,1,1 };  // Length of the tissue in x, y, and z [cm]
	float layerHeight = 1.0f; // the z-location where we change element height
	float layerSize = 2; // The number of elements corresponding to the first layer height
	float TC = 0; // Thermal Conductivity [W/cm C]
	float VHC = 0; // Volumetric Heat Capacity [W/cm^3]
	float MUA = 0; // Absorption Coefficient [cm^-1]
	float ambientTemp = 0;  // Temperature surrounding the tissue for Convection [C]
	Eigen::VectorXf Temp; // Our values for temperature at the nodes of the elements
	Eigen::VectorXf NFR; // Our values for Heat addition
	float alpha = 0.5; // time step weight
	float deltaT = 0.01; // time step [s]
	float tFinal = 1; // total duration of simulation [s]
	float Qn = 0; // heat escaping the Neumann Boundary
	float HTC = 1; // convective heat transfer coefficient [W/cm^2]
	int Nn1d = 2;
	bool elemNFR = false; // whether the NFR pertains to an element or a node
	std::vector<boundaryCond> boundaryType = { HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK }; // Individual boundary type for each face: 0: heat sink. 1: Flux Boundary. 2: Convective Boundary
	std::vector< std::array<float, 3 >> tempSensorLocations;
	std::vector<std::vector<float>> sensorTemps;

	FEM_Simulator() = default;
	FEM_Simulator(std::vector<std::vector<std::vector<float>>> Temp, float tissueSize[3], float TC, float VHC, float MUA, float HTC, int Nn1d=2);
	FEM_Simulator(FEM_Simulator& inputSim);
	void performTimeStepping();
	void createKMFelem();
	void createFirr();
	void updateTemperatureSensors(int timeIdx, Eigen::VectorXf& dVec);
	void setTemp(std::vector<std::vector<std::vector<float>>> Temp);
	void setTemp(Eigen::VectorXf& Temp);
	std::vector<std::vector<std::vector<float>>> getTemp();
	void setNFR(std::vector<std::vector<std::vector<float>>> NFR);
	void setNFR(Eigen::VectorXf& NFR);
	void setNFR(float laserPose[6], float laserPower, float beamWaist);
	void setTissueSize(float tissueSize[3]);
	void setLayer(float layerHeight, int layerSize);
	void setTC(float TC);
	void setVHC(float VHC);
	void setMUA(float MUA);
	void setHTC(float HTC);
	void setFlux(float Qn);
	void setAmbientTemp(float ambientTemp);
	void setGridSize(int gridSize[3]);
	void setNodeSize(int nodeSize[3]);
	void setSensorLocations(std::vector<std::array<float, 3>>& tempSensorLocations);
	void setJ(int layer=1);
	void setKeInt();
	void setMe();
	void setFeIrr();
	void setFeQ();
	void setFeConv();
	void setKeConv();
	void setBoundaryConditions(int BC[6]);
	Eigen::VectorXf getSensorTemps();

	/**********************************************************************************************************************/
	/***************	 These were all private but I made them public so I could unit test them **************************/

	// The Kint, M, and Firr matrices for the entire domain
	// Kint = Kint*kappa + Kconv*h
	Eigen::SparseMatrix<float, Eigen::RowMajor> Kint; // Conductivity matrix for non-dirichlet nodes
	Eigen::SparseMatrix<float, Eigen::RowMajor> Kconv; //Conductivity matrix due to convection
	Eigen::SparseMatrix<float, Eigen::RowMajor> M; // Row Major because we fill it in one row at a time for nodal build -- elemental it doesn't matter
	// Firr = Firr*muA + Fconv*h + Fk*kappa + Fq
	Eigen::VectorXf Firr; // forcing function due to irradiance
	Eigen::VectorXf Fconv; // forcing functino due to convection
	Eigen::VectorXf Fk; // forcing function due conductivity matrix on dirichlet nodes
	Eigen::VectorXf Fq; // forcing function due to constant flux boundary

	// because of our assumptions, these don't need to be recalculated every time and can be class variables.
	Eigen::MatrixXf KeInt; // Elemental Construction of Kint
	Eigen::MatrixXf Me; // Elemental construction of M
	Eigen::MatrixXf FeIrr; // Elemental Construction of Firr
	// FeQ is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	Eigen::MatrixXf FeQ; // Element Construction of Fq
	// FeConv is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	Eigen::MatrixXf FeConv; // Elemental Construction of FConv
	// KeConv is a 4x4 matrix for each face, but we save it as a vector of 8x8 matrices so we can take advantage of having local node coordinates A 
	std::array<Eigen::MatrixXf,6> KeConv; // Elemental construction of KConv
	Eigen::Matrix3<float> J = Eigen::Matrix3f::Constant(0.0f);
	Eigen::Matrix2<float> Js1 = Eigen::Matrix2f::Constant(0.0f);
	Eigen::Matrix2<float> Js2 = Eigen::Matrix2f::Constant(0.0f);
	Eigen::Matrix2<float> Js3 = Eigen::Matrix2f::Constant(0.0f);
	std::vector<int> validNodes; // global indicies on non-dirichlet boundary nodes
	std::vector<int> dirichletNodes;
	// this vector contains a mapping between the global node number and its index location in the reduced matrix equations. 
	// A value of -1 at index i, indicates that global node i is a dirichlet node. 
	std::vector<int> nodeMap; 

	void initializeBoundaryNodes(); // function has test cases
	void initializeElementNodeSurfaceMap(); // function has test case for Nn1d = 2
	void initializeElementMatrices(int layer); 
	int determineNodeFace(int globalNode); // function has test cases
	float calculateNA(float xi[3], int Ai); // function has test cases
	float calculateNABase(float xi, int Ai); //function has test cases
	Eigen::Matrix3<float> calculateJ(int layer=1); // function has test cases
	Eigen::Matrix2<float> calculateJs(int dim,int layer=1); // function has test cases
	float calculateNADotBase(float xi, int Ai); // function has test cases
	// function has test cases
	Eigen::Vector3<float> calculateNA_dot(float xi[3], int Ai);
	float integrate(float (FEM_Simulator::*func)(float[3], int, int), int points, int dim, int Ai, int Bi); // function has test cases
	void getGlobalPosition(int globalNode, float position[3]); // function not used because of uniform cuboid assumptions
	float calcKintAB(float xi[3], int Ai, int Bi); // function has test cases
	float calcMAB(float xi[3], int Ai, int Bi); // function has test cases
	float calcFintAB(float xi[3], int Ai, int Bi);
	float calcFqA(float xi[3], int Ai, int dim);
	float calcFconvA(float xi[3], int Ai, int dim);
	float calcKconvAB(float xi[3], int Ai, int dim);

	void ind2sub(int index, int size[3], int sub[3]); // function has test cases
};


