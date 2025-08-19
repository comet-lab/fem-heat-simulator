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

#ifdef USE_AMGX
#include "AmgXSolver.hpp"
#include <cuda_runtime.h>
#endif

class FEM_Simulator
{
public:
	bool silentMode = false; // controls print statements

	/* not used because we are no longer using arbitrary elements*/
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

	//0 - heat sink, 1 - heatFlux boundary, 2 - convection boundary
	enum boundaryCond { HEATSINK, FLUX, CONVECTION };

	FEM_Simulator();
	FEM_Simulator(std::vector<std::vector<std::vector<float>>> Temp, float tissueSize[3], float TC, float VHC, float MUA, float HTC, int Nn1d=2);
	FEM_Simulator(const FEM_Simulator& inputSim);
	void multiStep(float duration); // simulates multiple steps of time integration
	void singleStep(); // simulates a single step of time integration
	void buildMatrices(); // creates global matrices and performs spatial discretization
	void createFirr(); // creates only the Forcing vector for the fluence rate
	void applyParameters();
	void initializeTimeIntegration();
	void initializeModel();
	void initializeSensorTemps(int numSteps); // initialize sensor temps vec with 0s
	void updateTemperatureSensors(int timeIdx); // update sensor temp vec
	std::array<int, 3> positionToElement(std::array<float, 3>& position, float xi[3]); // Convert a 3D position into an element that contains that position
	

	// Setters and Getters
	void setTemp(std::vector<std::vector<std::vector<float>>> Temp);
	void setTemp(Eigen::VectorXf& Temp);
	std::vector<std::vector<std::vector<float>>> getTemp();
	void setFluenceRate(std::vector<std::vector<std::vector<float>>> FluenceRate);
	void setFluenceRate(Eigen::VectorXf& FluenceRate);
	void setFluenceRate(float laserPose[6], float laserPower, float beamWaist);
	void setFluenceRate(Eigen::Vector<float,6> laserPose, float laserPower, float beamWaist);
	void setTissueSize(float tissueSize[3]);
	void setDeltaT(float deltaT);
	void setLayer(float layerHeight, int elemsInLayer);
	void setTC(float TC);
	void setVHC(float VHC);
	void setMUA(float MUA);
	void setHTC(float HTC);
	void setFlux(float heatFlux);
	void setAmbientTemp(float ambientTemp);
	void setElementsPerAxis(int elementsPerAxis[3]);
	void setNodesPerAxis(int nodesPerAxis[3]);
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
	int elementsPerAxis[3] = { 1,1,1 }; // Number of elements in x, y, and z [voxels]
	int nodesPerAxis[3] = { 2,2,2 }; // Number of nodes in x, y, and z. Should be elementsPerAxis + 1;
	float tissueSize[3] = { 1,1,1 };  // Length of the tissue in x, y, and z [cm]
	float layerHeight = 1.0f; // the z-location where we change element height
	float elemsInLayer = 2; // The number of elements corresponding to the first layer height
	float TC = 0; // Thermal Conductivity [W/cm C]
	float VHC = 0; // Volumetric Heat Capacity [W/cm^3]
	float MUA = 0; // Absorption Coefficient [cm^-1]
	float HTC = 1; // convective heat transfer coefficient [W/cm^2]
	float ambientTemp = 0;  // Temperature surrounding the tissue for Convection [C]
	Eigen::VectorXf Temp; // Our values for temperature at the nodes of the elements
	Eigen::VectorXf dVec; // This is our discrete temperature at non-dirichlet nodes from Time-stepping
	Eigen::VectorXf vVec; // This is our discrete temperature velocity from Time-stepping
	Eigen::VectorXf FluenceRate; // Our values for Heat addition
	float alpha = 0.5; // time step weight
	float deltaT = 0.01; // time step [s]
	float heatFlux = 0; // heat escaping the Neumann Boundary
	
	int Nn1d = 2; // in a single element, the number of nodes used in one dimension
	bool parameterUpdate = true;
	bool fluenceUpdate = true;
	bool elemNFR = false; // whether the FluenceRate pertains to an element or a node
	std::vector<boundaryCond> boundaryType = { HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK }; // Individual boundary type for each face: 0: heat sink. 1: Flux Boundary. 2: Convective Boundary
	std::vector< std::array<float, 3 >> tempSensorLocations; // locations of temperature sensors
	std::vector<std::vector<float>> sensorTemps; // stored temperature information for each sensor over time. 

	/* This section of variables stores all the matrices used for building the first order ODE
	There are both global and elemental matrices specified. The global matrices are also distinguished
	by how they are created (e.g. conduction, laser, etc). These are saved as class attributes because
	the current build assumes them to be relatively constant throughout the mesh so its easier to save once*/
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cgSolver;
#ifdef USE_AMGX
	AmgXSolver* amgxSolver = nullptr;
#endif
	bool useGPU = false;

	Eigen::SparseMatrix<float> LHS; // this stores the left hand side of our matrix inversion, so the solver doesn't lose the reference.
	Eigen::SparseMatrix<float, Eigen::RowMajor> Kint; // Conductivity matrix for non-dirichlet nodes
	Eigen::SparseMatrix<float, Eigen::RowMajor> Kconv; //Conductivity matrix due to convection
	Eigen::SparseMatrix<float, Eigen::RowMajor> M; // Row Major because we fill it in one row at a time for nodal build -- elemental it doesn't matter
	
	Eigen::SparseMatrix<float, Eigen::RowMajor> FirrMat; // forcing function due to irradiance when multiplied by nodal fluence rate
	Eigen::VectorXf FirrElem; // forcing function due to irradiance when using elemental fluence rate
	Eigen::VectorXf Fconv; // forcing functino due to convection
	Eigen::VectorXf Fk; // forcing function due conductivity matrix on dirichlet nodes
	Eigen::VectorXf Fq; // forcing function due to constant heatFlux boundary

	// because of our assumptions, these don't need to be recalculated every time and can be class variables.
	Eigen::MatrixXf KeInt; // Elemental Construction of Kint
	Eigen::MatrixXf Me; // Elemental construction of M
	Eigen::MatrixXf FeIrr; // Elemental Construction of FirrElem
	// FeQ is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	Eigen::MatrixXf FeQ; // Element Construction of Fq
	// FeConv is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	Eigen::MatrixXf FeConv; // Elemental Construction of FConv
	// KeConv is a 4x4 matrix for each face, but we save it as a vector of 8x8 matrices so we can take advantage of having local node coordinates A 
	std::array<Eigen::MatrixXf,6> KeConv; // Elemental construction of KConv

	Eigen::SparseMatrix<float, Eigen::RowMajor> globK; // Global Conductivity Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor> globM; // Global Thermal Mass Matrix
	Eigen::VectorXf globF; // Global Heat Source Matrix 

	Eigen::Matrix3f J = Eigen::Matrix3f::Constant(0.0f); // bi-unit jacobian
	Eigen::Matrix2f Js1 = Eigen::Matrix2f::Constant(0.0f); // surface jacobian for yz plane
	Eigen::Matrix2f Js2 = Eigen::Matrix2f::Constant(0.0f); // surface jacobian for xz plane
	Eigen::Matrix2f Js3 = Eigen::Matrix2f::Constant(0.0f); // surface jacobian for xy plane.

	/* This section of variables stores vectors that help us distinguish what type of node or what
	kind of surface on the element we are at. They are built once during matrix creation */
	std::vector<int> validNodes; // global indicies on non-dirichlet boundary nodes
	std::vector<int> dirichletNodes;
	// this vector contains a mapping between the global node number and its index location in the reduced matrix equations. 
	// A value of -1 at index i, indicates that global node i is a dirichlet node. 
	std::vector<int> nodeMap; 
	// This maps the face on an element to the local node numbers on that face: top,bot,front,right,back,left
	std::array<std::vector<int>, 6> elemNodeSurfaceMap;

	void initializeBoundaryNodes(); // goes through each node and labels them if they are on the boundary
	void initializeElementNodeSurfaceMap(); // For an arbitrary element, maps what nodes could belong to which faces of the cuboid
	void initializeElementMatrices(int layer);  // sets the Ke, Me, etc matrices 
	int determineNodeFace(int globalNode); // determines if/which faces the global node is on
	float calculateNA(float xi[3], int Ai); // Calculates output of shape function 
	float calculateNABase(float xi, int Ai); // Calculates output of 1D shape function
	Eigen::Matrix3f calculateJ(int layer=1); // calculates volume bi-unit Jacobian
	Eigen::Matrix2f calculateJs(int dim,int layer=1); // calculates surface bi-unit Jacobian for each of 3 faces. 
	float calculateNADotBase(float xi, int Ai); // calculates output of 1D shape function derivative
	Eigen::Vector3f calculateNA_dot(float xi[3], int Ai); // calculates shape function derivative
	float integrate(float (FEM_Simulator::*func)(float[3], int, int), int points, int dim, int Ai, int Bi); // performs numerical integration
	void getGlobalPosition(int globalNode, float position[3]); // function not used because of uniform cuboid assumptions
	float calcKintAB(float xi[3], int Ai, int Bi); // function that is integrated to form Ke
	float calcMAB(float xi[3], int Ai, int Bi); // function that is integrated to form Me
	float calcFintAB(float xi[3], int Ai, int Bi); // function that is integrated to create Fe int
	float calcFqA(float xi[3], int Ai, int dim); // function that is integrated for Fq
	float calcFconvA(float xi[3], int Ai, int dim); // function that is integrated for Fconv
	float calcKconvAB(float xi[3], int Ai, int dim); // function that is integrated for Kconv

	void ind2sub(int index, int size[3], int sub[3]); 
	std::chrono::steady_clock::time_point printDuration(const std::string& message, std::chrono::steady_clock::time_point startTime);
	bool gpuAvailable();
};


