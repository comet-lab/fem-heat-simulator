#pragma once
#include <vector> 
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class FEM_Simulator
{
public:
	struct element {             // Structure declaration
		int elementNumber;         // Member (int variable)
		float globalNodePositions[8][3];   // Member (string variable)
	};       // Structure variable
	static const int A[8][3] = { {-1, -1, -1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1} };

	// One node can belong to multiple faces so we use binary to set the face value
	// 0 - internal node, 1 - top face, 2 - bottom face, 4 - front face, 8 - right fce, 16 - back face, 32 - left face
	enum tissueFace { INTERNAL = 0, TOP = 1, BOTTOM = 2, FRONT = 4, RIGHT = 8, BACK = 16, LEFT = 32 };
	//0 - heat sink, 1 - flux boundary, 2 - convection boundary
	enum boundaryCond { HEATSINK, FLUX, CONVECTION };

	int gridSize[3] = { 0,0,0 }; // Number of voxels in x, y, and z [voxels]
	int tissueSize[3] = { 0,0,0 };  // Length of the tissue in x, y, and z [cm]
	float TC = 0; // Thermal Conductivity [W/cm C]
	float VHC = 0; // Volumetric Heat Capacity [W/cm^3]
	float MUA = 0; // Absorption Coefficient [cm^-1]
	float ambientTemp = 0;  // Temperature surrounding the tissue for Convection [C]
	std::vector<std::vector<std::vector<float>>> Temp; // Our values for temperature at the nodes of the elements
	float alpha = 0.5; // time step weight
	float deltaT = 0.01; // time step [s]
	float tSpan[2] = { 0, 0 };
	float Jn = 0; // heat escaping the Neumann Boundary
	float HTC = 0; // convective heat transfer coefficient [W/cm^2]
	std::vector<boundaryCond> boundaryType = { HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK, HEATSINK }; // Individual boundary type for each face: 0: heat sink. 1: Flux Boundary. 2: Convective Boundary

	FEM_Simulator() = default;

	FEM_Simulator(std::vector<std::vector<std::vector<float>>> Temp, int tissueSize[3], float TC, float VHC, float MUA);

	void setBoundaryConditions(int BC[6]);

	void solveFEA(std::vector<std::vector<std::vector<float>>> NFR);

private:


	
	int nodeSize[3] = { 1,1,1 }; // Number of nodes in x, y, and z. Should be gridSize + 1;
	std::vector<int> nodeMapping; // Maps d values in the full d vector to the reduced d vector. It helps in creating the matrices already in reduced form
	//int currAi = 0;
	//int currBi = 0;
	element currElement;

	std::vector<int> dirichletBoundaryNodes;
	std::vector<int> fluxBoundaryNodes;
	std::vector<int> convectionBoundaryNodes;
	std::vector<int> bottomFaceNodes;
	std::vector<int> frontFaceNodes;
	std::vector<int> topFaceNodes;
	std::vector<int> rightFaceNodes;
	std::vector<int> backFaceNodes;
	std::vector<int> leftFaceNodes;

	void initializeBoundaryNodes();
	int determineNodeFace(int globalNode);
	bool checkBoundaryNode(int globalNode);
	void initializeNodeMap();
	float calculateNA(float xi[3], int Ai);
	static void calculateJ(float xi[3], float deltaX[3], Eigen::Matrix3<float> J);
	static void calculateJs(float xi[2], float  pos[8][3], int dim, Eigen::Matrix2<float> Js);
	static void calculateNA_dot(float xi[3], int Ai, Eigen::Vector3<float> NA_dot);
	static float calculateNA_xi(float xi[3], int Ai);
	static float calculateNA_eta(float xi[3], int Ai);
	static float calculateNA_zeta(float xi[3], int Ai);
	static void ind2sub(int index, int size[3], int sub[3]);
	float integrate(std::function<float(float[3], int Ai, int Bi)> fun, int points, int dim, int Ai, int Bi);
	void getGlobalNodesFromElem(int elem, int nodes[8]);
	void getGlobalPosition(int globalNode, float position[3]);
	float calculateKAB(float xi[3]);
	float createKABFunction(float xi[3], int Ai, int Bi);
	float createMABFunction(float xi[3], int Ai, int Bi);

};


