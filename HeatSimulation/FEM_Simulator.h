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
	const int dimMap[6] = { -3, 3, -2, 1, 2, -1 }; // top is actually negative z axis... a bit confusing

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
	float Jn = 0; // heat escaping the Neumann Boundary
	float HTC = 1; // convective heat transfer coefficient [W/cm^2]
	int Nn1d = 2;
	bool elemNFR = false; // whether the NFR pertains to an element or a node
	std::vector<boundaryCond> boundaryType = { HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK }; // Individual boundary type for each face: 0: heat sink. 1: Flux Boundary. 2: Convective Boundary
	std::vector< std::array<float, 3 >> tempSensorLocations;
	std::vector<std::vector<float>> sensorTemps;

	FEM_Simulator() = default;
	FEM_Simulator(std::vector<std::vector<std::vector<float>>> Temp, float tissueSize[3], float TC, float VHC, float MUA, float HTC, int Nn1d=2);
	void performTimeStepping();
	void createKMF();
	void createKMFelem();
	void updateTemperatureSensors(int timeIdx, Eigen::VectorXf& dVec);
	void setTemp(std::vector<std::vector<std::vector<float>>> Temp);
	void setTemp(Eigen::VectorXf& Temp, int gridSize[3]);
	void setNFR(std::vector<std::vector<std::vector<float>>> NFR);
	void setNFR(Eigen::VectorXf& NFR);
	void setTissueSize(float tissueSize[3]);
	void setLayer(float layerHeight, int layerSize);
	void setTC(float TC);
	void setVHC(float VHC);
	void setMUA(float MUA);
	void setHTC(float HTC);
	void setJn(float Jn);
	void setAmbientTemp(float ambientTemp);
	void setGridSize(int gridSize[3]);
	void setNodeSize(int nodeSize[3]);
	void setSensorLocations(std::vector<std::array<float, 3>>& tempSensorLocations);
	void setJ(int layer=1);
	void setKe();
	void setKn();
	void setMe();
	void setMn();
	void setFeInt();
	void setFnInt();
	void setFj();
	void setFv();
	void setFvu();
	void setBoundaryConditions(int BC[6]);

	/**********************************************************************************************************************/
	/***************	 These were all private but I made them public so I could unit test them **************************/

	// The K, M, and F matrices for the entire domain
	Eigen::SparseMatrix<float, Eigen::RowMajor> K; // Row Major because we fill it in one row at a time for nodal build -- elemental it doesn't matter
	Eigen::SparseMatrix<float, Eigen::RowMajor> M; // Row Major because we fill it in one row at a time for nodal build -- elemental it doesn't matter
	Eigen::VectorXf F;

	// because of our assumptions, these don't need to be recalculated every time and can be class variables.
	Eigen::MatrixXf Ke; // Elemental Construction of K
	Eigen::MatrixXf Me; // Elemental construction of M
	Eigen::MatrixXf FeInt; // Elemental Construction of F_int
	// Fje is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	Eigen::MatrixXf Fje; 
	// Fve is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	Eigen::MatrixXf Fve;
	std::array<Eigen::MatrixXf,6> Kje; // Kje is a 4x4 matrix for each face, but we save it as a vector of 8x8 matrices so we can take advantage of having local node coordinates A 
	Eigen::Matrix3<float> J = Eigen::Matrix3f::Constant(0.0f);
	Eigen::Matrix2<float> Js1 = Eigen::Matrix2f::Constant(0.0f);
	Eigen::Matrix2<float> Js2 = Eigen::Matrix2f::Constant(0.0f);
	Eigen::Matrix2<float> Js3 = Eigen::Matrix2f::Constant(0.0f);
	std::vector<int> validNodes; // global indicies on non-dirichlet boundary nodes
	std::vector<int> dirichletNodes;
	// this vector contains a mapping between the global node number and its index location in the reduced matrix equations. 
	// A value of -1 at index i, indicates that global node i is a dirichlet node. 
	std::vector<int> nodeMap; 
	element currElement;

	void initializeBoundaryNodes();
	void initializeElementNodeSurfaceMap();
	int determineNodeFace(int globalNode); // function has test cases
	float calculateNA(float xi[3], int Ai); // function has test cases
	float calculateNABase(float xi, int Ai);
	Eigen::Matrix3<float> calculateJ(int layer=1); // function has test cases
	Eigen::Matrix2<float> calculateJs(int dim,int layer=1);
	float calculateNADotBase(float xi, int Ai);
	void getGlobalNodesFromElem(int elem, int nodes[8]);
	// function has test cases
	Eigen::Vector3<float> calculateNA_dot(float xi[3], int Ai);
	float integrate(float (FEM_Simulator::*func)(float[3], int, int), int points, int dim, int Ai, int Bi); // function has test cases
	void getGlobalPosition(int globalNode, float position[3]); // function not used because of uniform cuboid assumptions
	Eigen::Vector<int,27> getNodeNeighbors(int globalNode);
	std::vector<int> convertToLocalNode(int globalNode,int f);
	int convertToGlobalNode(int localNode,int globalReference,int localReference);
	int convertToNeighborIdx(int globalNode, int globalReference);
	float createKABFunction(float xi[3], int Ai, int Bi); // function has test cases
	float createMABFunction(float xi[3], int Ai, int Bi); // function has test cases
	float createFintFunction(float xi[3], int Ai, int Bi);
	float createFjFunction(float xi[3], int Ai, int dim);
	float createFvFunction(float xi[3], int Ai, int dim);
	float createFvuFunction(float xi[3], int Ai, int dim);

	void ind2sub(int index, int size[3], int sub[3]); // function has test cases
	static void reduceSparseMatrix(Eigen::SparseMatrix<float> oldMat, std::vector<int> rowsToRemove, Eigen::SparseMatrix<float>* newMat, Eigen::SparseMatrix<float> *suppMat, int nNodes);
	

};


