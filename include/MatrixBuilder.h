#pragma once
#include <vector> 
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <stdexcept>

/*static const std::array<std::array<int, 3>, 8> A_HEX_LIN = { { {-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1} } };
static const std::array<std::array<int, 3>, 4> A_TET_LIN = { { {0,0,0},{1,0,0},{0,1,0},{0,0,1} } };
static const std::array<std::array<int, 3>, 10> A_TET_QUAD = { { {0,0,0},{1,0,0},{0,1,0},{0,0,1},{0.5,0,0},{0.5,0.5,0},{0,0.5,0},{0,0,0.5},{0.5,0,0.5},{0,0.5,0.5} } };*/
enum BoundaryType {
	NONE,
	HEATSINK,
	FLUX,
	CONVECTION
};

struct Node {
	double x;
	double y;
	double z;
	bool isDirichlet = false;
};

struct Element {
	std::vector<long> nodes; // index in node list used in element
	std::array<BoundaryType, 6> faceBoundary = { NONE,NONE,NONE,NONE,NONE,NONE };
};

enum GeometricOrder {
	LINEAR,
	QUADRATIC
};

enum Shape {
	TETRAHEDRAL,
	HEXAHEDRAL
};

class MatrixBuilder
{
public:


	MatrixBuilder();
	MatrixBuilder(std::vector<Node> nodeList, std::vector<Element> elemList);
	MatrixBuilder(std::string filename);

	void resetMatrices();
	void buildMatrices();
	void applyElement(Element elem, long elemIdx);
	void applyBoundary(Element elem);
	float calculateHexFunction1D(float xi, int A);
	float calculateHexFunction3D(const std::array<float, 3>& xi, int A);
	float calculateHexFunctionDeriv1D(float xi, int A);
	Eigen::Vector3f calculateHexFunctionDeriv3D(const std::array<float, 3>& xi, int A);
	float calculateTetFunction3D(const std::array<float, 3>& xi, int A);
	Eigen::Vector3f calculateTetFunctionDeriv3D(const std::array<float, 3>& xi, int A);

	Eigen::MatrixXf calculateKe(const Element& elem);
	Eigen::MatrixXf calculateFeFlux(const Element& elem, int faceIndex, float q);
	Eigen::MatrixXf calculateFeConv(const Element& elem, int faceIndex);
	Eigen::MatrixXf calculateMe(Element elem);
	Eigen::Matrix3f calculateJ(const Element& elem, const std::array<float, 3>& xi);
	std::array<long, 3> ind2sub(long idx, const std::array<long, 3>& size);
	void calculateJs(const Element& elem, int face);
	



	void setMesh(std::vector<Node> nodeList, std::vector<Element> elemList, std::vector<BoundaryType> boundary);
	void setNodeMap();
	void setNodeList(std::vector<Node> nodeList);
	void setElementList(std::vector<Element> elemList);
	void setBoundary(std::vector<BoundaryType> boundary);
	template <class F>
	void integrateHex8(const Element& elem, bool needDeriv, F&& body);
	template <class F>
	void integrateHexFace4(const Element& elem, int faceIndex, F&& body);

	std::vector<Node> nodeList() { return nodeList_; }
	std::vector<Element> elemList() { return elemList_; }
	std::vector<BoundaryType> boundary() { return boundary_; }
	std::vector<long> nodeMap() { return nodeMap_; }
	GeometricOrder order() { return order_; }

private:
	std::vector<Node> nodeList_;
	std::vector<Element> elemList_;
	std::vector<BoundaryType> boundary_;
	GeometricOrder order_ = LINEAR;
	int nN1D_ = 2;
	Shape elementShape_ = HEXAHEDRAL;
	// this vector contains a mapping between the global node number and its index location in the reduced matrix equations. 
	// A value of -1 at index i, indicates that global node i is a dirichlet node. 
	std::vector<long> nodeMap_;
	// contains the global indicies of every non-dirichlet node
	std::vector<long> validNodes_;

	//Eigen::Matrix3f J_; // Jacobian of our current element

	Eigen::SparseMatrix<float, Eigen::RowMajor> M_; // Thermal Mass Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor> K_; // Thermal Conductivity Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor> Q_; // Convection Matrix -- should have the same structure as Me just gets scaled by htc instead of vhc

	// Internal nodal heat generation (aka laser). Its size is nNodes x nNodes. Becomes a vector once post multiplied by a 
	// vector dictating the fluence experienced at each node. 
	Eigen::SparseMatrix<float, Eigen::RowMajor> Fint_; 
	// forcing function due to irradiance when using elemental fluence rate. Its size is nNodes x nElems. Becomes vector once post multiplied
	// by a vector dictating the fluence experienced by each element
	Eigen::SparseMatrix<float, Eigen::RowMajor> FintElem_; 
	// forcing function due to convection on dirichlet node. Size is nNodes x nNodes. Becomes a vector once post multiplied
	// by a vector specifying the fixed temperature at each element.
	Eigen::SparseMatrix<float, Eigen::RowMajor> Fconv_;
	// Forcing Function due to conductivity matrix on dirichlet nodes. Stored as a matrix but becomes vector once multiplied by nodal temperatures
	Eigen::SparseMatrix<float, Eigen::RowMajor> Fk_; 
	Eigen::VectorXf Fflux_; // forcing function due to constant heatFlux boundary
	Eigen::VectorXf Fq_; // forcing function due to ambient temperature

	//// because of our assumptions, these don't need to be recalculated every time and can be class variables.
	//Eigen::MatrixXf Ke_; // Elemental Construction of Kint
	//Eigen::MatrixXf Me_; // Elemental construction of M
	//Eigen::MatrixXf FeInt_; // Elemental Construction of FirrElem
	//// FeQ is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	//Eigen::MatrixXf FeQ_; // Element Construction of Fq
	//// FeConv is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	//Eigen::MatrixXf FeConv_; // Elemental Construction of FConv
	//// KeConv is a 4x4 matrix for each face, but we save it as a vector of 8x8 matrices so we can take advantage of having local node coordinates A 
	//std::array<Eigen::MatrixXf, 6> Qe_; // Elemental construction of KConv


};


template <class F>
inline void MatrixBuilder::integrateHex8(const Element& elem, bool needDeriv, F&& body)
{
	const float g = 1.0f / std::sqrt(3.0f);
	const float gp[8][3] = {
		{-g,-g,-g}, { g,-g,-g}, { g, g,-g}, {-g, g,-g},
		{-g,-g, g}, { g,-g, g}, { g, g, g}, {-g, g, g}
	};

	for (int k = 0; k < 8; k++)
	{
		// Parent-domain coordinates
		std::array<float,3> xi = { gp[k][0], gp[k][1], gp[k][2] };

		// Compute shape data
		float N[8];
		Eigen::Vector3f dN_dxi[8];

		for (int A = 0; A < 8; A++)
		{
			N[A] = calculateHexFunction3D(xi, A);
			if (needDeriv)
				dN_dxi[A] = calculateHexFunctionDeriv3D(xi, A);
		}

		// Compute Jacobian using your dedicated function
		Eigen::Matrix3f J = calculateJ(elem, xi);       // fills J_

		float detJ = J.determinant();
		if (detJ <= 0.0f)
			throw std::runtime_error("Negative Jacobian in integrateHex8().");


		// Compute derivatives in physical space
		Eigen::Vector3f dN_dx[8];
		if (needDeriv)
		{
			Eigen::Matrix3f Jinv = J.inverse();		
			for (int a = 0; a < 8; ++a)
				dN_dx[a] = Jinv * dN_dxi[a];  // chain rule: dN/dx = J^-1 * dN/dxi
		}

		// All Gaussian weights = 1, so weight = detJ
		float weight = detJ;

		// Call user integrand
		body(N, dN_dx, weight);
	}
}


template <class F>
inline void MatrixBuilder::integrateHexFace4(const Element& elem, int faceIndex, F&& body)
{
	// Hex8 face local node indices
	constexpr int faceNodes[6][4] = {
		{0,1,3,2}, {4,5,6,7}, {0,1,5,4},
		{3,2,6,7}, {0,2,6,4}, {1,3,7,5}
	};
	const int* nodesOnFace = faceNodes[faceIndex];
	//std::cout << "Nodes on Face: " << nodesOnFace[0] << ", " << nodesOnFace[1] << ", " << nodesOnFace[2] << ", " << nodesOnFace[3] << std::endl;

	const float g = 1.0f / std::sqrt(3.0f);
	float gp[2] = { -g, g }; // 2-point Gauss quadrature in each parametric direction

	// Loop over 2x2 Gauss points
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
		{
			float xi_face[2] = { gp[i], gp[j] };

			// Map 2D face coordinates to 3D element parent coordinates
			std::array<float,3> xi;
			switch (faceIndex)
			{
			case 0: xi[0] = xi_face[0]; xi[1] = xi_face[1]; xi[2] = -1.0f; break; // bottom
			case 1: xi[0] = xi_face[0]; xi[1] = xi_face[1]; xi[2] = 1.0f; break; // top
			case 2: xi[0] = xi_face[0]; xi[1] = -1.0f;   xi[2] = xi_face[1]; break; // front
			case 3: xi[0] = xi_face[0]; xi[1] = 1.0f;   xi[2] = xi_face[1]; break; // back
			case 4: xi[0] = -1.0f;   xi[1] = xi_face[0]; xi[2] = xi_face[1]; break; // left
			case 5: xi[0] = 1.0f;   xi[1] = xi_face[0]; xi[2] = xi_face[1]; break; // right
			}

			// Compute shape functions and derivatives for the 4 face nodes
			float N_face[4];
			Eigen::Vector3f dN_dxi[4];
			for (int a = 0; a < 4; ++a)
			{
				int nodeIdx = nodesOnFace[a];
				N_face[a] = calculateHexFunction3D(xi, nodeIdx);
				dN_dxi[a] = calculateHexFunctionDeriv3D(xi, nodeIdx);
			}

			// Compute Jacobian for the face
			Eigen::Matrix3f J = calculateJ(elem, xi);
			Eigen::Vector3f t1, t2;
			switch (faceIndex)
			{
			case 0: t1 = J.col(0); t2 = J.col(1); break; // bot face
			case 1: t1 = J.col(0); t2 = J.col(1); break; // top face
			case 2: t1 = J.col(0); t2 = J.col(2); break; // front face
			case 3: t1 = J.col(0); t2 = J.col(2); break; // back face
			case 4: t1 = J.col(1); t2 = J.col(2); break; // left face
			case 5: t1 = J.col(1); t2 = J.col(2); break; // right face
			}
			float dS = t1.cross(t2).norm();

			float weight = dS; // 2x2 Gauss weights = 1*1

			// Call user integrand
			body(N_face, nodesOnFace, weight);
		}
}