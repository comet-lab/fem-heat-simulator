#pragma once
#include <vector> 
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>

class FEM_Simulator
{
public:
	struct element {             // Structure declaration
		int elementNumber;         // Member (int variable)
		float globalNodePositions[8][3];   // Member (string variable)
	};       // Structure variable
	static const int A[8][3];

	// One node can belong to multiple faces so we use binary to set the face value
	// 0 - internal node, 1 - top face, 2 - bottom face, 4 - front face, 8 - right fce, 16 - back face, 32 - left face
	enum tissueFace { INTERNAL = 0, TOP = 1, BOTTOM = 2, FRONT = 4, RIGHT = 8, BACK = 16, LEFT = 32 };
	// This maps the face to the axis number and direction of the face: top, bottom, front, right, back, left
	const int dimMap[6] = { 3, -3, -2, 1, 2, -1 };
	// This maps the face on an element to the local node numbers on that face: top,bot,front,right,back,left
	const int elemNodeSurfaceMap[6][4] = { {4,5,6,7},{0,1,2,3},{0,1,4,5},{1,2,5,6},{2,3,6,7},{0,3,4,7} };
	//0 - heat sink, 1 - flux boundary, 2 - convection boundary
	enum boundaryCond { HEATSINK, FLUX, CONVECTION };

	int gridSize[3] = { 1,1,1 }; // Number of voxels in x, y, and z [voxels]
	float tissueSize[3] = { 1,1,1 };  // Length of the tissue in x, y, and z [cm]
	float TC = 0; // Thermal Conductivity [W/cm C]
	float VHC = 0; // Volumetric Heat Capacity [W/cm^3]
	float MUA = 0; // Absorption Coefficient [cm^-1]
	float ambientTemp = 0;  // Temperature surrounding the tissue for Convection [C]
	std::vector<std::vector<std::vector<float>>> Temp; // Our values for temperature at the nodes of the elements
	//std::vector<std::vector<std::vector<float>>> NFR; // Our values for Heat addition
	float alpha = 0.5; // time step weight
	float deltaT = 0.01; // time step [s]
	float tFinal = 1; // total duration of simulation [s]
	float Jn = 0; // heat escaping the Neumann Boundary
	float HTC = 1; // convective heat transfer coefficient [W/cm^2]
	std::vector<boundaryCond> boundaryType = { HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK }; // Individual boundary type for each face: 0: heat sink. 1: Flux Boundary. 2: Convective Boundary

	FEM_Simulator() = default;
	FEM_Simulator(std::vector<std::vector<std::vector<float>>> Temp, float tissueSize[3], float TC, float VHC, float MUA, float HTC);
	void solveFEA(std::vector<std::vector<std::vector<float>>> NFR);
	void setInitialTemperature(std::vector<std::vector<std::vector<float>>> Temp);
	void setTissueSize(float tissueSize[3]);
	void setTC(float TC);
	void setVHC(float VHC);
	void setMUA(float MUA);
	void setHTC(float HTC);
	void setJn(float Jn);
	void setAmbientTemp(float ambientTemp);
	void setGridSize(int gridSize[3]);
	void setNodeSize(int nodeSize[3]);
	void setJ();
	void setKe();
	void setMe();
	void setFeInt();
	void setFj();
	void setFv();
	void setFvu();
	void setBoundaryConditions(int BC[6]);

	/**********************************************************************************************************************/
	/***************	 These were all private but I made them public so I could unit test them **************************/

	// because of our assumptions, these don't need to be recalculated every time and can be class variables.
	Eigen::Matrix<float,8,8> Ke = Eigen::Matrix<float,8,8>::Constant(0.0f);
	Eigen::Matrix<float,8,8> Me = Eigen::Matrix<float, 8, 8>::Constant(0.0f);
	Eigen::Matrix<float, 8, 8> FeInt = Eigen::Matrix<float, 8, 8>::Constant(0.0f);
	// Fj is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	Eigen::Matrix<float, 8, 6> Fj = Eigen::Matrix<float, 8, 6>::Constant(0.0f); 
	// Fv is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	Eigen::Matrix<float, 8, 6> Fv = Eigen::Matrix<float, 8, 6>::Constant(0.0f);
	std::vector<Eigen::Matrix<float, 8, 8>> Fvu{ 6, Eigen::Matrix<float, 8, 8>::Constant(0.0f) }; // Fvu is a 4x4 matrix for each face, but we save it as a vector of 8x8 matrices so we can take advantage of having A 
	Eigen::Matrix3<float> J = Eigen::Matrix3f::Constant(0.0f);
	Eigen::Matrix2<float> Js1 = Eigen::Matrix2f::Constant(0.0f);
	Eigen::Matrix2<float> Js2 = Eigen::Matrix2f::Constant(0.0f);
	Eigen::Matrix2<float> Js3 = Eigen::Matrix2f::Constant(0.0f);
	int nodeSize[3] = { 2,2,2 }; // Number of nodes in x, y, and z. Should be gridSize + 1;
	std::vector<int> validNodes; // global indicies on non-dirichlet boundary nodes
	std::vector<int> dirichletNodes;
	// this vector contains a mapping between the global node number and its index location in the reduced matrix equations. 
	// A value of -1 at index i, indicates that global node i is a dirichlet node. 
	std::vector<int> nodeMap; 
	element currElement;

	void initializeBoundaryNodes();
	int determineNodeFace(int globalNode); // function has test cases
	float calculateNA(float xi[3], int Ai); // function has test cases
	Eigen::Matrix3<float> calculateJ(); // function has test cases
	Eigen::Matrix2<float> calculateJs(int dim); // function has test cases
	static Eigen::Vector3<float> calculateNA_dot(float xi[3], int Ai);
	static float calculateNA_xi(float xi[3], int Ai); // function has test cases
	static float calculateNA_eta(float xi[3], int Ai); // function has test cases
	static float calculateNA_zeta(float xi[3], int Ai); // function has test cases
	float integrate(float (FEM_Simulator::*func)(float[3], int, int), int points, int dim, int Ai, int Bi); // function has test cases
	void getGlobalNodesFromElem(int elem, int nodes[8]); // function has test cases
	void getGlobalPosition(int globalNode, float position[3]); // function not used because of uniform cuboid assumptions
	float createKABFunction(float xi[3], int Ai, int Bi); // function has test cases
	float createMABFunction(float xi[3], int Ai, int Bi); // function has test cases
	float createFintFunction(float xi[3], int Ai, int Bi);
	float createFjFunction(float xi[3], int Ai, int dim);
	float createFvFunction(float xi[3], int Ai, int dim);
	float createFvuFunction(float xi[3], int Ai, int dim);

	static void ind2sub(int index, int size[3], int sub[3]); // function has test cases
	static void reduceSparseMatrix(Eigen::SparseMatrix<float> oldMat, std::vector<int> rowsToRemove, Eigen::SparseMatrix<float>* newMat, Eigen::SparseMatrix<float> *suppMat, int nNodes);
	

};


