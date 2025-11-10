#pragma once
#include <vector> 
#include <algorithm>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <stdexcept>

struct Node {
	double x;
	double y;
	double z;
	BoundaryType boundary = BoundaryType::NONE;
};

struct Element {
	std::vector<Node> nodes;
};

enum BoundaryType {
	NONE,
	HEATSINK, 
	FLUX, 
	CONVECTION
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
	void MatrixBuilder() Default;
	void MatrixBuilder(std::vector<Node> nodeList, std::vector<Element> elemList);
	void MatrixBuilder(std::string filename);

	void resetMatrices();
	void buildMatrices();
	void calculateHexFunction1D(float xi, int A);
	void calculateHexFunction3D(float[3] xi, int A);
	void calculateShapeFunctionDeriv1D(float xi, int A);
	void calculateShapeFunctionDeriv3D(float[3] xi, int A);
	void calculateKe(Element elem);
	void calculateMe(Element elem);
	void calculateJ(Element elem);
	void calculateJs(Element elem, int face);
	void calculateQe(Element elem);


	
	void setMesh(std::vector<Node> nodeList, std::vector<Element> elemList, std::vector<BoundaryType> boundary);
	void setNodeMap();
	void setNodeList(std::vector<Node> nodeList);
	void setElementList(std::vector<Element> elemList);
	void setBoundary(std::vector<BoundaryType> boundary);
	


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
	std::vector<long> validNodes;

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