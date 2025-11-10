#pragma once
#include <vector> 
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <stdexcept>


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
	BoundaryType boundary = BoundaryType::NONE;
};

struct Element {
	std::vector<Node> nodes;
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
	float calculateHexFunction1D(float xi, int A);
	float calculateHexFunction3D(const std::array<float, 3>& xi, int A);
	float calculateHexFunctionDeriv1D(float xi, int A);
	Eigen::Vector3f calculateHexFunctionDeriv3D(const std::array<float, 3>& xi, int A);
	Eigen::Matrix<float, 8, 8> calculateKe(const Element& elem);
	Eigen::Matrix<float, 8, 8> calculateMe(Element elem);
	void calculateJ(const Element& elem, const std::array<float, 3>& xi);
	std::array<long, 3> ind2sub(long idx, const std::array<long, 3>& size);
	void calculateJs(const Element& elem, int face);
	void calculateQe(const Element& elem);



	void setMesh(std::vector<Node> nodeList, std::vector<Element> elemList, std::vector<BoundaryType> boundary);
	void setNodeMap();
	void setNodeList(std::vector<Node> nodeList);
	void setElementList(std::vector<Element> elemList);
	void setBoundary(std::vector<BoundaryType> boundary);
	template <class F>
	void integrateHex8(const Element& elem, F&& body);

	std::vector<Node> nodeList() { return nodeList_; }
	std::vector<Element> elemList() { return elemList_; }
	std::vector<BoundaryType> boundary() { return boundary_; }
	std::vector<long> nodeMap() { return nodeMap_; }
	GeometricOrder order() { return order_; }

	Eigen::Matrix<float, 3, 3> J() { return J_; };


private:
	std::vector<Node> nodeList_;
	std::vector<Element> elemList_;
	std::vector<BoundaryType> boundary_;
	GeometricOrder order_ = LINEAR;
	int nN1D_ = 2;
	Shape elementShape_ = HEXAHEDRAL;
	//std::vector<long> validNodes_; // global indicies on non-dirichlet boundary nodes
	// this vector contains a mapping between the global node number and its index location in the reduced matrix equations. 
	// A value of -1 at index i, indicates that global node i is a dirichlet node. 
	std::vector<long> nodeMap_;
	std::vector<long> validNodes_;

	Eigen::Matrix3f J_; // Jacobian of our current element

	Eigen::SparseMatrix<float, Eigen::RowMajor> M_; // Thermal Mass Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor> K_; // Thermal Conductivity Matrix
	Eigen::SparseMatrix<float, Eigen::RowMajor> Q_; // Convection Matrix

	Eigen::SparseMatrix<float, Eigen::RowMajor> Fint_; // Internal nodal heat generation (aka laser)
	Eigen::VectorXf FirrElem_; // forcing function due to irradiance when using elemental fluence rate
	Eigen::VectorXf Fconv_; // forcing functino due to convection
	Eigen::VectorXf Fk_; // Forcing Function due to conductivity matrix on dirichlet nodes
	Eigen::VectorXf Fq_; // forcing function due to constant heatFlux boundary

	// because of our assumptions, these don't need to be recalculated every time and can be class variables.
	Eigen::MatrixXf Ke_; // Elemental Construction of Kint
	Eigen::MatrixXf Me_; // Elemental construction of M
	Eigen::MatrixXf FeInt_; // Elemental Construction of FirrElem
	// FeQ is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	Eigen::MatrixXf FeQ_; // Element Construction of Fq
	// FeConv is a 4x1 vector for each face, but we save it as an 8x6 matrix so we can take advantage of having A
	Eigen::MatrixXf FeConv_; // Elemental Construction of FConv
	// KeConv is a 4x4 matrix for each face, but we save it as a vector of 8x8 matrices so we can take advantage of having local node coordinates A 
	std::array<Eigen::MatrixXf, 6> Qe_; // Elemental construction of KConv


};


template <class F>
inline void MatrixBuilder::integrateHex8(const Element& elem, F&& body)
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
			dN_dxi[A] = calculateHexFunctionDeriv3D(xi, A);
		}

		// Compute Jacobian using your dedicated function
		calculateJ(elem, xi);       // fills J_

		float detJ = J_.determinant();
		if (detJ <= 0.0f)
			throw std::runtime_error("Negative Jacobian in integrateHex8().");

		// All Gaussian weights = 1, so weight = detJ
		float weight = detJ;

		// Call user integrand
		body(N, dN_dxi, weight);
	}
}
